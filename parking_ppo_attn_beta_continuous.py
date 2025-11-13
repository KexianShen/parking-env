# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

import parking_env


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=123296,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Parking-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1024,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=6,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(
            id=env_id,
            render_mode="no_render",
            observation_type="vector",
            action_type="continuous",
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.FrameStack(env, 8)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(p=dropout)

        # Linear transformations for queries, keys, and values
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        num_batch = query.size(0)

        # Linearly transform query, key, and value
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape Q, K, and V for multi-head attention
        Q = Q.view(num_batch, -1, self.num_heads, self.head_dim)
        K = K.view(num_batch, -1, self.num_heads, self.head_dim)
        V = V.view(num_batch, -1, self.num_heads, self.head_dim)

        # Transpose to prepare for batch-wise matrix multiplication
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Compute scaled dot-product attention scores
        scores = torch.einsum("bhid,bhjd->bhij", Q, K) / self.head_dim**0.5

        # Apply attention mask if needed (e.g., for masking padding tokens)
        if mask is not None:
            scores.masked_fill_(mask == 0, float("-inf"))

        # Apply softmax to obtain attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Compute the weighted sum using einsum
        weighted_sum = torch.einsum("bhij,bhjd->bhid", attn_weights, V)

        # Reshape and concatenate the heads
        weighted_sum = (
            weighted_sum.permute(0, 2, 1, 3)
            .contiguous()
            .view(num_batch, -1, self.num_heads * self.head_dim)
        )

        # Linearly transform the concatenated heads
        output = self.W_o(weighted_sum)

        return output, attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=128, ff_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_attn = MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm1 = nn.BatchNorm1d(num_heads)
        self.feedforward1 = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.BatchNorm1d(num_heads)
        self.dropout = nn.Dropout(p=dropout)

        self.social_attn = MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm3 = nn.BatchNorm1d(num_heads)
        self.feedforward2 = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )
        self.norm4 = nn.BatchNorm1d(num_heads)

    def forward(self, query: Tensor):
        num_batch = query.shape[0]
        num_agent = query.shape[1]
        num_time = query.shape[2]

        # Temporal-attention
        query = query.view(num_batch * num_agent, num_time, self.embed_dim)
        attn_output, _ = self.temporal_attn(query=query, key=query, value=query)
        query = query + self.dropout(attn_output)
        query = self.norm1(query)
        ff_output = self.feedforward1(query)
        query = query + self.dropout(ff_output)
        query = self.norm2(query)

        # Social-attention
        query = query.view(num_batch, num_agent * num_time, self.embed_dim)
        attn_output, _ = self.social_attn(
            query=query[:, 0:num_time, :],
            key=query[:, num_time:, :],
            value=query[:, num_time:, :],
        )
        query = query[:, 0:num_time, :] + self.dropout(attn_output)
        query = self.norm3(query)
        ff_output = self.feedforward2(query)
        query = query + self.dropout(ff_output)
        query = self.norm4(query)

        query = query.view(num_batch, 1, num_time, self.embed_dim)
        return query


class Encoder(nn.Module):
    def __init__(
        self,
        proj_cfg: dict = dict(input_dim=13, hidden_dim=512, output_dim=128),
        attn_cfg: dict = dict(embed_dim=128, ff_dim=256, num_heads=8, dropout=0.1),
    ):
        super().__init__()
        self.proj_cfg = proj_cfg
        self.attn_cfg = attn_cfg
        self._init_layers()

    def _init_layers(self):
        self.proj_layer = MLP(**self.proj_cfg)
        self.attn_layer = EncoderLayer(**self.attn_cfg)

    def pre_encoder(self, feats: Tensor):
        def pos_encodings(num_pos, encoding_dim):
            position = torch.arange(0, num_pos, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, encoding_dim, 2, dtype=torch.float32)
                * -(np.log(10000.0) / encoding_dim)
            )
            encodings = torch.zeros(1, num_pos, encoding_dim)
            encodings[0, :, 0::2] = torch.sin(position * div_term)
            encodings[0, :, 1::2] = torch.cos(position * div_term)
            return encodings

        assert feats.dim() == 4, "obs should be NxAxTxD"
        num_batch = feats.shape[0]
        num_agent = feats.shape[1]
        num_time = feats.shape[2]
        feats = feats.view(num_batch * num_agent, num_time, self.attn_cfg["embed_dim"])
        pos_enc = (
            pos_encodings(num_pos=num_time, encoding_dim=self.attn_cfg["embed_dim"])
            .expand(num_batch, num_agent, -1, -1)
            .view(-1, num_time, self.attn_cfg["embed_dim"])
            .to(feats.device)
        )
        feats = feats + pos_enc
        return feats.view(num_batch, num_agent, num_time, self.attn_cfg["embed_dim"])

    def forward(self, feats: Tensor):
        feats = self.proj_layer(feats)
        feats = self.pre_encoder(feats)
        feats = self.attn_layer(feats)
        feats = feats.flatten(1)
        return feats


class Agent(nn.Module):
    def __init__(
        self,
        proj_cfg: dict = dict(input_dim=13, hidden_dim=512, output_dim=128),
        attn_cfg: dict = dict(embed_dim=128, ff_dim=256, num_heads=8, dropout=0.1),
    ) -> None:
        super().__init__()
        self.proj_cfg = proj_cfg
        self.attn_cfg = attn_cfg
        self._init_layers()

    def _init_layers(self):
        self.encoder = Encoder(self.proj_cfg, self.attn_cfg)
        decoder_input_dim = self.attn_cfg["embed_dim"] * self.attn_cfg["num_heads"]
        self.critic = MLP(input_dim=decoder_input_dim, hidden_dim=64, output_dim=1)
        self.actor_alpha = MLP(input_dim=decoder_input_dim, hidden_dim=64, output_dim=1)
        self.actor_beta = MLP(input_dim=decoder_input_dim, hidden_dim=64, output_dim=1)

    def get_value(self, obs: Tensor):
        return self.critic(self.encoder(obs.permute(0, 2, 1, 3)))

    def get_action_and_value(self, obs: Tensor, action=None):
        feats = self.encoder(obs.permute(0, 2, 1, 3))
        action_alpha = torch.exp(self.actor_alpha(feats)).clamp(max=5)
        action_beta = torch.exp(self.actor_beta(feats)).clamp(max=5)
        probs = Beta(action_alpha, action_beta)
        if action is None:
            action = 2 * probs.sample() - 1
        epsilon = 1e-5
        action = torch.clamp(action, min=-1.0 + epsilon, max=1 - epsilon)
        return (
            action,
            probs.log_prob((action + 1) / 2).sum(1),
            probs.entropy().sum(1),
            self.critic(feats),
        )


def kaiming_init_hook(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = Agent().apply(kaiming_init_hook).to(device)
    # try:
    #     agent = torch.load("ppo_attn_beta_continuous.pth")
    # except:
    #     pass
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy().astype(np.float64)
            )
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(done).to(device),
            )

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()

    torch.save(agent, "ppo_attn_beta_continuous.pth")
