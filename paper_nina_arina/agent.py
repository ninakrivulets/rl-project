import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor, Critic


class SRSACAgent:
    def __init__(self, specs, config, device):
        self.specs = specs
        self.config = config
        self.device = device
        self.action_low = torch.tensor(specs["action_low"], device=device)
        self.action_high = torch.tensor(specs["action_high"], device=device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        self.target_entropy = config["target_entropy"]
        if self.target_entropy is None:
            self.target_entropy = -specs["action_dim"] / 2
        self._build()

    def _build(self):
        hidden_dims = self.config["hidden_dims"]
        obs_dim = self.specs["obs_dim"]
        action_dim = self.specs["action_dim"]
        self.actor = Actor(
            obs_dim,
            action_dim,
            hidden_dims,
            self.config["log_std_min"],
            self.config["log_std_max"],
        ).to(self.device)
        self.critic1 = Critic(obs_dim, action_dim, hidden_dims).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic1 = Critic(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic2 = Critic(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config["actor_lr"])
        self.critic1_opt = torch.optim.Adam(
            self.critic1.parameters(), lr=self.config["critic_lr"]
        )
        self.critic2_opt = torch.optim.Adam(
            self.critic2.parameters(), lr=self.config["critic_lr"]
        )

        self.learnable_temperature = self.config.get("learnable_temperature", True)
        if self.learnable_temperature:
            init_temperature = float(self.config["init_temperature"])
            self.log_alpha = torch.tensor(
                np.log(init_temperature), device=self.device, requires_grad=True
            )
            self.alpha_opt = torch.optim.Adam(
                [self.log_alpha], lr=self.config["alpha_lr"]
            )
        else:
            self.log_alpha = torch.tensor(
                np.log(float(self.config["init_temperature"])),
                device=self.device,
            )
            self.alpha_opt = None

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def reset(self):
        self._build()

    def _to_tensor(self, array):
        return torch.tensor(array, dtype=torch.float32, device=self.device)

    def _rescale_action(self, normalized_action):
        return normalized_action * self.action_scale + self.action_bias

    def sample_action(self, obs, explore=True):
        obs = self._to_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, _, mu = self.actor.sample(obs)
            chosen = action if explore else mu
            env_action = self._rescale_action(chosen)
        clipped = torch.max(torch.min(env_action, self.action_high), self.action_low)
        return clipped.cpu().numpy()[0]

    def update(self, batch):
        obs = self._to_tensor(batch["obs"])
        actions = self._to_tensor(batch["actions"])
        rewards = self._to_tensor(batch["rewards"])
        next_obs = self._to_tensor(batch["next_obs"])
        dones = self._to_tensor(batch["dones"])

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            next_action = self._rescale_action(next_action)
            target_q1 = self.target_critic1(next_obs, next_action)
            target_q2 = self.target_critic2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target = rewards + self.config["discount"] * (1 - dones) * target_q

        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)

        self.critic1_opt.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_opt.step()

        sampled_action, log_prob, _ = self.actor.sample(obs)
        env_action = self._rescale_action(sampled_action)
        q1_pi = self.critic1(obs, env_action)
        q2_pi = self.critic2(obs, env_action)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_pi, q2_pi)).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.learnable_temperature:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

        self._soft_update(self.critic1, self.target_critic1, self.config["tau"])
        self._soft_update(self.critic2, self.target_critic2, self.config["tau"])

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
        }

    def state_dict(self):
        payload = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "config": self.config,
        }
        if self.alpha_opt is not None:
            payload["alpha_opt"] = self.alpha_opt.state_dict()
        return payload

    def load_state_dict(self, payload):
        self.actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.target_critic1.load_state_dict(payload["target_critic1"])
        self.target_critic2.load_state_dict(payload["target_critic2"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic1_opt.load_state_dict(payload["critic1_opt"])
        self.critic2_opt.load_state_dict(payload["critic2_opt"])
        self.log_alpha = payload["log_alpha"].to(self.device).requires_grad_(
            self.learnable_temperature
        )
        if self.alpha_opt is not None and "alpha_opt" in payload:
            self.alpha_opt.load_state_dict(payload["alpha_opt"])

    def _soft_update(self, source, target, tau):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

