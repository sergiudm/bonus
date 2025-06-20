from collections import deque
import os 

import numpy as np
import matplotlib.pyplot as plt
import torch
from agent import DQNAgent, ReinforceAgent, A2CAgent, DDPGAgent
from env import CourseSelectionEnv
import time


def plot_all_scores(all_scores):
    """将所有算法的训练曲线绘制在同一张图上进行对比。"""
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)

    for algo_name, scores in all_scores.items():
        # 计算并绘制百次滑动平均分，这比原始分更能体现趋势
        rolling_avg = [np.mean(scores[i-100:i]) for i in range(100, len(scores))]
        plt.plot(np.arange(100, len(scores)), rolling_avg, label=algo_name)

    plt.ylabel('100-Episode Average Reward')
    plt.xlabel('Episode #')
    plt.title('Comparison of RL Algorithms on Course Selection Task')
    plt.legend()
    plt.grid(True)
    plt.savefig('assets/comparison_chart5.png', dpi=300)

def show_policy(agent, env, action_space=None, is_continuous=False, output_dir='results'):
    """
    通用策略展示函数，将结果以Markdown表格形式保存到文件中。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    agent_type = type(agent).__name__
    filepath = os.path.join(output_dir, f"{agent_type}_policy.md")
    
    # 使用一个列表来缓存Markdown输出
    md_output = []

    # 1. 添加主标题和描述信息
    md_output.append(f"# {agent_type} 最优选课策略\n\n")
    md_output.append("该文件展示了经过训练的Agent在面对所有课程时，所做出的最优出价决策。\n\n")
    md_output.append("### Agent的课程偏好分布\n")
    md_output.append(f"- **最喜欢**: `{env.preferences['most_liked']}`\n")
    md_output.append(f"- **中等喜好**: `{env.preferences['medium_liked']}`\n")
    md_output.append(f"- **不喜欢**: `{env.preferences['disliked']}`\n\n")
    
    # 2. 创建Markdown表格的表头
    md_output.append("### 详细决策过程\n\n")
    md_output.append("| 课程 ID | 课程类型 | 决策时剩余积分 | Agent决策 (出价) |\n")
    md_output.append("|:---:|:---:|:---:|:---:|\n")
    
    # 将模型设置为评估模式
    if hasattr(agent, 'qnetwork_local'): agent.qnetwork_local.eval()
    if hasattr(agent, 'policy'): agent.policy.eval()
    if hasattr(agent, 'actor'): agent.actor.eval()
    if hasattr(agent, 'actor_local'): agent.actor_local.eval()

    state = env.reset()
    with torch.no_grad():
        for i in range(env.num_courses):
            if is_continuous:
                bid_amount = agent.act(state, noise=0)[0] 
            else:
                if agent_type in ['ReinforceAgent', 'A2CAgent']:
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
                    probs = agent.policy(state_tensor) if hasattr(agent, 'policy') else agent.actor(state_tensor)
                    action_index = torch.argmax(probs).item()
                else:
                    action_index = agent.act(state, eps=0.0)
                bid_amount = action_space[action_index]

            if bid_amount > env.remaining_points: bid_amount = 0
            
            # 准备表格行所需的数据
            course_id = env.current_course_index
            remaining_points_before_bid = env.remaining_points
            pref_type = env.pref_map[course_id]
            pref_map_str = {2: "最喜欢", 1: "中等喜好", 0: "不喜欢"}
            bid_str = f"{bid_amount:.2f}" if is_continuous else f"{bid_amount}"
            
            # 3. 将决策结果格式化为Markdown表格的一行并添加到列表中
            md_output.append(f"| {course_id} | {pref_map_str[pref_type]} | {remaining_points_before_bid} | **{bid_str}** |\n")
            
            # 更新状态
            state, _, _, _ = env.step(bid_amount)
            
    # 将模型恢复为训练模式
    if hasattr(agent, 'qnetwork_local'): agent.qnetwork_local.train()
    if hasattr(agent, 'policy'): agent.policy.train()
    if hasattr(agent, 'actor'): agent.actor.train()
    if hasattr(agent, 'actor_local'): agent.actor_local.train()

    # 4. 将列表中的所有内容写入文件
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(md_output)
        print(f"策略已保存至: {filepath}")
    except IOError as e:
        print(f"错误：无法写入文件 {filepath}。原因: {e}")


### --- Training Loops (Unchanged) ---
def train_dqn(agent, env, action_space, n_episodes):
    scores, scores_window = [], deque(maxlen=100)
    eps = 1.0
    episode_buffer = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        episode_buffer.clear()
        while True:
            action_index = agent.act(state, eps); action_value = action_space[action_index]
            next_state, reward, done, _ = env.step(action_value)
            episode_buffer.append((state, action_index, next_state, done))
            state = next_state
            if done:
                for s, a_idx, ns, d in episode_buffer: agent.step(s, a_idx, reward, ns, d)
                scores_window.append(reward); scores.append(reward)
                break
        eps = max(0.01, 0.998 * eps)
        if i_episode % 100 == 0: print(f'\rEpisode {i_episode}/{n_episodes}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode == n_episodes: print(f'\rEpisode {i_episode}/{n_episodes}\tAverage Score: {np.mean(scores_window):.2f}')
    return scores
    
def train_policy_based(agent, env, action_space, n_episodes): # For REINFORCE and A2C
    scores, scores_window = [], deque(maxlen=100)
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        while True:
            action_index = agent.act(state); action_value = action_space[action_index]
            state, reward, done, _ = env.step(action_value)
            if done:
                agent.learn(reward); scores_window.append(reward); scores.append(reward)
                break
        if i_episode % 100 == 0: print(f'\rEpisode {i_episode}/{n_episodes}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode == n_episodes: print(f'\rEpisode {i_episode}/{n_episodes}\tAverage Score: {np.mean(scores_window):.2f}')
    return scores

def train_ddpg(agent, env, n_episodes):
    scores, scores_window = [], deque(maxlen=100)
    episode_buffer = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        episode_buffer.clear()
        while True:
            action_value = agent.act(state, noise=0.3)[0]
            next_state, reward, done, _ = env.step(action_value)
            episode_buffer.append((state, action_value, next_state, done))
            state = next_state
            if done:
                for s, a_val, ns, d in episode_buffer: agent.step(s, a_val, reward, ns, d)
                scores_window.append(reward); scores.append(reward)
                break
        if i_episode % 100 == 0: print(f'\rEpisode {i_episode}/{n_episodes}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode == n_episodes: print(f'\rEpisode {i_episode}/{n_episodes}\tAverage Score: {np.mean(scores_window):.2f}')
    return scores

# ==============================================================================
# Part 4: Main Execution Block (Modified for comparison)
# ==============================================================================
if __name__ == '__main__':
    # 统一训练参数
    N_EPISODES = 6000
    N_EPISODES_REINFORCE = 6000 

    device='cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    
    # 存储所有结果
    all_scores = {}
    trained_agents = {}
    
    # 初始化通用环境和动作空间
    env = CourseSelectionEnv()
    state_size = env.get_state_size()
    discrete_action_space = [0, 5, 10, 15, 20, 30, 40, 50]
    continuous_max_action_value = 50.0

    print("using device:", device)

    # --- 1. 训练 DQN ---
    print("\n--- Training DQN ---")
    start_time = time.time()
    dqn_agent = DQNAgent(state_size=state_size, action_space=discrete_action_space, seed=0, device=device)
    dqn_scores = train_dqn(dqn_agent, env, discrete_action_space, n_episodes=N_EPISODES)
    all_scores['DQN'] = dqn_scores
    trained_agents['DQN'] = dqn_agent
    print(f"\nDQN Training Time: {time.time() - start_time:.2f} seconds")
    torch.save(dqn_agent.qnetwork_local.state_dict(), 'ckpt/dqn_agent.pth')

    # --- 2. 训练 REINFORCE ---
    print("\n--- Training REINFORCE ---")
    start_time = time.time()
    reinforce_agent = ReinforceAgent(state_size=state_size, action_space=discrete_action_space, seed=0, device=device)
    reinforce_scores = train_policy_based(reinforce_agent, env, discrete_action_space, n_episodes=N_EPISODES_REINFORCE)
    all_scores['REINFORCE'] = reinforce_scores
    trained_agents['REINFORCE'] = reinforce_agent
    print(f"\nREINFORCE Training Time: {time.time() - start_time:.2f} seconds")
    torch.save(reinforce_agent.policy.state_dict(), 'ckpt/reinforce_agent.pth')

    # --- 3. 训练 A2C ---
    print("\n--- Training A2C ---")
    start_time = time.time()
    a2c_agent = A2CAgent(state_size=state_size, action_space=discrete_action_space, seed=0, device=device)
    a2c_scores = train_policy_based(a2c_agent, env, discrete_action_space, n_episodes=N_EPISODES)
    all_scores['A2C'] = a2c_scores
    trained_agents['A2C'] = a2c_agent
    print(f"\nA2C Training Time: {time.time() - start_time:.2f} seconds")
    torch.save(a2c_agent.actor.state_dict(), 'ckpt/a2c_agent_actor.pth')
    torch.save(a2c_agent.critic.state_dict(), 'ckpt/a2c_agent_critic.pth')

    # --- 4. 训练 DDPG ---
    print("\n--- Training DDPG ---")
    start_time = time.time()
    ddpg_agent = DDPGAgent(state_size=state_size, max_action_value=continuous_max_action_value, seed=0, device=device)
    ddpg_scores = train_ddpg(ddpg_agent, env, n_episodes=N_EPISODES)
    all_scores['DDPG'] = ddpg_scores
    trained_agents['DDPG'] = ddpg_agent
    print(f"\nDDPG Training Time: {time.time() - start_time:.2f} seconds")
    # torch.save(ddpg_agent.actor_local.state_dict(), 'ckpt/ddpg_actor.pth')
    # torch.save(ddpg_agent.critic_local.state_dict(), 'ckpt/ddpg_critic.pth')
    # torch.save(ddpg_agent.actor_target.state_dict(), 'ckpt/ddpg_actor_target.pth')
    # torch.save(ddpg_agent.critic_target.state_dict(), 'ckpt/ddpg_critic_target.pth')
    
    # --- 5. 绘制对比图 ---
    print("\n--- Plotting Comparison Chart ---")
    plot_all_scores(all_scores)

    # --- 6. 展示所有学到的策略 ---
    print("\n--- Displaying Learned Policies ---")
    show_policy(trained_agents['DQN'], env, action_space=discrete_action_space)
    show_policy(trained_agents['REINFORCE'], env, action_space=discrete_action_space)
    show_policy(trained_agents['A2C'], env, action_space=discrete_action_space)
    show_policy(trained_agents['DDPG'], env, is_continuous=True)