# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 20:59:32 2026

@author: admin
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class MoonColonyOptimizer:
    def __init__(self):
        # 参数定义（来自您的文档）
        self.M_total = 100000000  # 1亿公吨
        self.C_elevator = 179000  # 太空电梯年运力 17.9万吨/年
        self.c_elevator = 500000  # 太空电梯单位成本 50万美元/吨
        self.c_rocket = 80000000  # 火箭单次发射成本 8000万美元/次
        self.Y_max = 50  # 单发射场年发射上限
        
        # 发射场参数（低:4个, 中:3个, 高:3个）
        self.low_lat_q = 145  # 低纬度载重
        self.mid_lat_q = 125  # 中纬度载重  
        self.high_lat_q = 105  # 高纬度载重
        
    def pure_elevator_scenario(self):
        """纯太空电梯方案计算"""
        T = self.M_total / self.C_elevator
        cost = self.c_elevator * self.M_total
        emission = 0
        return cost, T, emission
    
    def pure_rocket_scenario(self, num_low=4, num_mid=3, num_high=3):
        """纯火箭方案计算"""
        # 计算总运力
        total_capacity = (num_low * self.low_lat_q + 
                         num_mid * self.mid_lat_q + 
                         num_high * self.high_lat_q) * self.Y_max
        
        T = self.M_total / total_capacity
        
        # 平均载重计算
        avg_payload = (num_low * self.low_lat_q + num_mid * self.mid_lat_q + num_high * self.high_lat_q) / (num_low + num_mid + num_high)
        total_launches = self.M_total / avg_payload
        
        cost = total_launches * self.c_rocket
        emission = total_launches * 100  # 100吨CO2e/次
        
        return cost, T, emission
    
    def mixed_transport_objective(self, x):
        """混合运输目标函数"""
        elevator_ratio = x[0]  # 电梯运输比例
        rocket_ratio = 1 - elevator_ratio  # 火箭运输比例
        
        # 成本计算
        elevator_cost = elevator_ratio * self.M_total * self.c_elevator
        rocket_launches = (rocket_ratio * self.M_total) / 125  # 平均载重125吨
        rocket_cost = rocket_launches * self.c_rocket
        total_cost = elevator_cost + rocket_cost
        
        # 周期计算（取两者最大值）
        elevator_time = (elevator_ratio * self.M_total) / self.C_elevator
        rocket_time = (rocket_ratio * self.M_total) / (10 * 50 * 125)  # 10个场，平均载重
        
        construction_time = max(elevator_time, rocket_time)
        
        # 多目标：最小化成本和周期（加权和）
        weight_cost = 0.6
        weight_time = 0.4
        
        return weight_cost * (total_cost / 1e12) + weight_time * construction_time
    
    def optimize_mixed_transport(self):
        """优化混合运输方案"""
        # 初始猜测：各50%
        x0 = [0.5]
        bounds = [(0.1, 0.9)]  # 电梯比例在10%-90%之间
        
        result = minimize(self.mixed_transport_objective, x0, 
                         method='SLSQP', bounds=bounds)
        
        if result.success:
            optimal_ratio = result.x[0]
            cost, time, emission = self.calculate_mixed_metrics(optimal_ratio)
            return optimal_ratio, cost, time, emission
        else:
            # 如果优化失败，使用默认值
            return 0.5, *self.calculate_mixed_metrics(0.5)
    
    def calculate_mixed_metrics(self, elevator_ratio):
        """计算混合方案的指标"""
        rocket_ratio = 1 - elevator_ratio
        
        # 成本
        elevator_cost = elevator_ratio * self.M_total * self.c_elevator
        rocket_launches = (rocket_ratio * self.M_total) / 125
        rocket_cost = rocket_launches * self.c_rocket
        total_cost = elevator_cost + rocket_cost
        
        # 周期
        elevator_time = (elevator_ratio * self.M_total) / self.C_elevator
        rocket_time = (rocket_ratio * self.M_total) / (10 * 50 * 125)
        construction_time = max(elevator_time, rocket_time)
        
        # 排放
        emission = rocket_launches * 100
        
        return total_cost, construction_time, emission



def main():
    optimizer = MoonColonyOptimizer()
    
    print("月球殖民地建设运输方案优化计算")
    print("=" * 50)
    
    # 计算纯方案
    elevator_cost, elevator_time, elevator_emission = optimizer.pure_elevator_scenario()
    rocket_cost, rocket_time, rocket_emission = optimizer.pure_rocket_scenario()
    
    # 优化混合方案
    optimal_ratio, mixed_cost, mixed_time, mixed_emission = optimizer.optimize_mixed_transport()
    
    # 输出结果表格
    print("\n=== 优化结果表格 ===")
    print("运输场景\t\t总成本(万亿美元)\t建设周期(年)\t年碳排放(万吨CO2e)")
    print("-" * 70)
    print(f"纯太空电梯\t\t{elevator_cost/1e12:.2f}\t\t{elevator_time:.1f}\t\t{elevator_emission/10000:.2f}")
    print(f"纯火箭\t\t\t{rocket_cost/1e12:.2f}\t\t{rocket_time:.1f}\t\t{rocket_emission/10000:.2f}")
    print(f"混合运输({optimal_ratio*100:.1f}:{100-optimal_ratio*100:.1f})\t{mixed_cost/1e12:.2f}\t\t{mixed_time:.1f}\t\t{mixed_emission/10000:.2f}")
    
    # 敏感性分析
    print("\n=== 敏感性分析 ===")
    ratios = [0.2, 0.4, 0.6, 0.8]
    for ratio in ratios:
        cost, time, emission = optimizer.calculate_mixed_metrics(ratio)
        print(f"电梯比例{ratio*100:.0f}%: 成本{cost/1e12:.2f}万亿美元, 周期{time:.1f}年, 排放{emission/10000:.2f}万吨")
    
    # 结果分析
    print("\n=== 结果分析与建议 ===")
    cost_saving_rocket = (rocket_cost - mixed_cost) / rocket_cost * 100
    time_saving_elevator = (elevator_time - mixed_time) / elevator_time * 100
    emission_saving = (rocket_emission - mixed_emission) / rocket_emission * 100
    
    print(f"混合方案相比纯火箭方案：")
    print(f"  • 成本节省: {cost_saving_rocket:.1f}%")
    print(f"  • 碳排放减少: {emission_saving:.1f}%")
    print(f"混合方案相比纯电梯方案：")
    print(f"  • 建设周期缩短: {time_saving_elevator:.1f}%")
    
    # 推荐方案
    print(f"\n推荐方案：太空电梯运输比例 {optimal_ratio*100:.1f}%，火箭运输比例 {(1-optimal_ratio)*100:.1f}%")
    print("该方案在成本、周期和环境影响之间取得最佳平衡")
    
    # 可视化结果
    plot_results(optimizer)
    
    # 输出文档表格格式
    print("\n=== 文档表格数据（直接复制到Word文档） ===")
    print("| 运输场景 | 总成本(万亿美元) | 建设周期(年) | 年碳排放(万吨CO2e) |")
    print("|---------|----------------|-------------|-------------------|")
    print(f"| 纯太空电梯 | {elevator_cost/1e12:.2f} | {elevator_time:.1f} | {elevator_emission/10000:.2f} |")
    print(f"| 纯火箭 | {rocket_cost/1e12:.2f} | {rocket_time:.1f} | {rocket_emission/10000:.2f} |")
    print(f"| 混合运输({optimal_ratio*100:.1f}:{100-optimal_ratio*100:.1f}) | {mixed_cost/1e12:.2f} | {mixed_time:.1f} | {mixed_emission/10000:.2f} |")

if __name__ == "__main__":
    main()