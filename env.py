import numpy as np
import random
from collections import defaultdict


class CourseSelectionEnv:
    """
    一个模拟学生选课的强化学习环境.
    Agent需要为b门课依次出积分.
    """
    def __init__(self, num_students=600, num_courses=25, x=4, y=8, z=8):
        self.num_students = num_students
        self.num_courses = num_courses
        self.course_capacity = 30
        self.min_courses_req = 3
        self.initial_points = 100

        # 定义我们Agent的课程偏好
        self.x, self.y, self.z = x, y, z
        all_course_ids = list(range(num_courses))
        random.shuffle(all_course_ids)
        self.preferences = {
            "most_liked": all_course_ids[:x],
            "medium_liked": all_course_ids[x:x+y],
            "disliked": all_course_ids[x+y:]
        }
        self.pref_map = {cid: 2 for cid in self.preferences["most_liked"]}
        self.pref_map.update({cid: 1 for cid in self.preferences["medium_liked"]})
        self.pref_map.update({cid: 0 for cid in self.preferences["disliked"]})

        # 为其他学生生成固定的随机出价策略
        self.other_students_bids = self._generate_other_bids()
        self.reset()

    def _generate_other_bids(self):
        """为其他99名学生生成模拟出价。"""
        bids = []
        for _ in range(self.num_students - 1):
            student_bids = {}
            points = self.initial_points
            num_to_bid = random.randint(self.min_courses_req, 7)
            courses_to_bid = random.sample(range(self.num_courses), num_to_bid)
            
            for course_id in courses_to_bid:
                if points > 0:
                    # 简单策略：在要选的课里随机分配分数
                    bid_amount = random.randint(1, max(1, int(points / (len(courses_to_bid) - len(student_bids)) if len(student_bids) < len(courses_to_bid) else 1)))
                    student_bids[course_id] = bid_amount
                    points -= bid_amount
            bids.append(student_bids)
        return bids

    def reset(self):
        """重置环境，开始新一轮选课。"""
        self.remaining_points = self.initial_points
        self.current_course_index = 0
        self.agent_bids = {}  # {course_id: bid_amount}
        self.history = [] # 存储(state, action_index)序列
        return self._get_state()

    def _get_state(self):
        """获取当前状态。"""
        pref_type = self.pref_map[self.current_course_index]
        pref_one_hot = np.zeros(3)
        pref_one_hot[pref_type] = 1.0

        state = np.concatenate([
            np.array([self.remaining_points / self.initial_points]),
            pref_one_hot,
            np.array([(self.num_courses - self.current_course_index) / self.num_courses])
        ])
        return state.astype(np.float32)

    def step(self, action_value):
        """执行一个动作（出价）。"""
        bid_amount = action_value
        
        if bid_amount > self.remaining_points:
            bid_amount = 0 # 如果出价超过剩余分数，则视为无效出价

        if bid_amount > 0:
            self.agent_bids[self.current_course_index] = bid_amount
        
        self.remaining_points -= bid_amount
        self.current_course_index += 1

        done = self.current_course_index == self.num_courses
        next_state = self._get_state() if not done else np.zeros(self.get_state_size())
        
        # 奖励只在最后计算，过程中奖励为0
        reward = 0
        if done:
            reward = self._calculate_final_reward()
            
        return next_state, reward, done, {}

    def _calculate_final_reward(self):
        """在一轮结束后，根据选课结果计算总奖励。"""
        all_bids = defaultdict(list)
        for course_id, bid in self.agent_bids.items():
            all_bids[course_id].append({"student_id": "agent", "bid": bid})
        for i, student_bids in enumerate(self.other_students_bids):
            for course_id, bid in student_bids.items():
                all_bids[course_id].append({"student_id": i, "bid": bid})

        agent_successful_courses = []
        for course_id, bids in all_bids.items():
            bids.sort(key=lambda x: x["bid"], reverse=True)
            top_bids = bids[:self.course_capacity]

            if len(bids) > self.course_capacity:
                cutoff_bid = top_bids[-1]["bid"]
                if bids[self.course_capacity]["bid"] == cutoff_bid:
                    top_bids = [b for b in top_bids if b["bid"] > cutoff_bid]

            if any(b["student_id"] == "agent" for b in top_bids):
                agent_successful_courses.append(course_id)

        # --- 计算奖励 ---
        if len(agent_successful_courses) < self.min_courses_req:
            return -100.0

        total_reward = 0.0
        for course_id in agent_successful_courses:
            pref_type = self.pref_map[course_id]
            if pref_type == 2: total_reward += 85.0 # 最喜欢
            elif pref_type == 1: total_reward += 17.0 # 中等
            else: total_reward += 9.0 # 不喜欢

        failed_bids_count = len(self.agent_bids) - len(agent_successful_courses)
        total_reward -= failed_bids_count * 2.0 # 惩罚浪费的积分

        return total_reward
    
    def get_state_size(self):
        return 1 + 3 + 1 # remaining_points_ratio, one_hot_pref, courses_left_ratio