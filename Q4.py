# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 23:29:57 2026

@author: admin
"""

"""
月球殖民地物流系统 - 修正的现实模型
使用更现实的参数
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

class RealisticLunarTransportModel:
    """现实的月球运输模型"""
    
    def __init__(self):
        # 基本参数
        self.M_total = 1e8  # 总运输量: 1亿吨建筑材料
        
        # 重新设计更现实的参数
        # 火箭参数
        self.rocket_payload = 20000  # 20吨有效载荷（更现实）
        self.num_launch_sites = 20  # 更多发射场
        self.launches_per_site_per_year = 20  # 更现实的发射频率
        self.rocket_cost_per_launch = 50e6  # 5000万美元/发射
        
        # 太空电梯参数
        self.elevator_capacity_per_year = 500000  # 50万吨/年（更现实）
        self.elevator_cost_per_ton = 200  # 200美元/吨
        
        # 计算容量
        self.max_rocket_launches_per_year = self.num_launch_sites * self.launches_per_site_per_year
        self.max_rocket_transport_per_year = self.max_rocket_launches_per_year * self.rocket_payload
        self.max_total_transport_per_year = self.elevator_capacity_per_year + self.max_rocket_transport_per_year
        
        # 计算最小建设时间
        self.min_construction_time = int(np.ceil(self.M_total / self.max_total_transport_per_year))
        self.max_construction_time = 30
        
        # 环境参数
        self.rocket_pollution_per_launch = 100  # 100吨污染物/发射
        
        print("="*60)
        print("REALISTIC LUNAR TRANSPORT MODEL")
        print("="*60)
        print(f"Total transport material: {self.M_total/1e6:.0f} million tons")
        
        print(f"\nTRANSPORT CAPACITIES:")
        print(f"Space Elevator:")
        print(f"  Annual capacity: {self.elevator_capacity_per_year/1e3:.1f} ktons/year")
        print(f"  Cost: ${self.elevator_cost_per_ton}/ton")
        
        print(f"\nRockets:")
        print(f"  Number of launch sites: {self.num_launch_sites}")
        print(f"  Launches per site per year: {self.launches_per_site_per_year}")
        print(f"  Payload per launch: {self.rocket_payload/1000:.0f} tons")
        print(f"  Cost per launch: ${self.rocket_cost_per_launch/1e6:.0f} million")
        print(f"  Max annual launches: {self.max_rocket_launches_per_year:,}")
        print(f"  Max annual transport: {self.max_rocket_transport_per_year/1e3:.1f} ktons")
        
        print(f"\nTOTAL CAPACITY:")
        print(f"  Max total transport per year: {self.max_total_transport_per_year/1e3:.1f} ktons")
        print(f"  Minimum construction time: {self.min_construction_time} years")
        print(f"  Maximum construction time: {self.max_construction_time} years")
        print("="*60)
    
    def calculate_feasible_solutions(self):
        """计算可行的解决方案"""
        solutions = []
        
        # 可能的建设时间
        possible_times = list(range(self.min_construction_time, self.max_construction_time + 1))
        
        print(f"\nCalculating solutions for construction times: {possible_times} years")
        
        for T in possible_times:
            # 年总需求
            annual_demand = self.M_total / T
            
            # 检查是否超过总容量
            if annual_demand > self.max_total_transport_per_year:
                continue
            
            # 尝试不同的电梯使用策略
            for elevator_share in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                # 计算电梯运输量
                annual_elevator_target = min(annual_demand * elevator_share, self.elevator_capacity_per_year)
                actual_elevator_share = annual_elevator_target / annual_demand
                
                # 计算火箭运输量
                annual_rocket_needed = max(0, annual_demand - annual_elevator_target)
                
                # 检查火箭容量
                if annual_rocket_needed > self.max_rocket_transport_per_year:
                    continue
                
                # 计算火箭发射次数
                annual_rocket_launches = annual_rocket_needed / self.rocket_payload
                
                # 检查发射场容量
                if annual_rocket_launches > self.max_rocket_launches_per_year:
                    continue
                
                # 计算总运输量
                total_elevator_transport = annual_elevator_target * T
                total_rocket_transport = annual_rocket_needed * T
                total_transport = total_elevator_transport + total_rocket_transport
                
                # 验证运输量
                transport_error = abs(total_transport - self.M_total) / self.M_total
                if transport_error > 0.01:  # 允许1%误差
                    continue
                
                # 计算目标值
                total_cost = (total_elevator_transport * self.elevator_cost_per_ton + 
                            annual_rocket_launches * T * self.rocket_cost_per_launch)
                
                environmental_impact = annual_rocket_launches * T * self.rocket_pollution_per_launch
                
                # 计算电梯比例
                elevator_ratio = actual_elevator_share * 100
                
                solutions.append({
                    'construction_time': T,
                    'elevator_share': elevator_share,
                    'total_cost': total_cost,
                    'environmental_impact': environmental_impact,
                    'elevator_transport': total_elevator_transport,
                    'rocket_transport': total_rocket_transport,
                    'total_transport': total_transport,
                    'annual_elevator': annual_elevator_target,
                    'annual_rocket': annual_rocket_needed,
                    'annual_rocket_launches': annual_rocket_launches,
                    'total_rocket_launches': annual_rocket_launches * T,
                    'elevator_ratio': elevator_ratio,
                    'annual_demand': annual_demand,
                    'transport_error': transport_error
                })
        
        return solutions
    
    def find_pareto_front(self, solutions):
        """找到帕累托前沿"""
        if not solutions or len(solutions) < 2:
            return solutions
        
        pareto_front = []
        
        for i, sol_i in enumerate(solutions):
            dominated = False
            
            for j, sol_j in enumerate(solutions):
                if i != j:
                    # 检查sol_j是否支配sol_i
                    if (sol_j['total_cost'] <= sol_i['total_cost'] and 
                        sol_j['construction_time'] <= sol_i['construction_time'] and 
                        sol_j['environmental_impact'] <= sol_i['environmental_impact'] and
                        (sol_j['total_cost'] < sol_i['total_cost'] or 
                         sol_j['construction_time'] < sol_i['construction_time'] or 
                         sol_j['environmental_impact'] < sol_i['environmental_impact'])):
                        dominated = True
                        break
            
            if not dominated:
                pareto_front.append(sol_i)
        
        return pareto_front
    
    def analyze_solutions(self, solutions):
        """分析解决方案"""
        if not solutions:
            print("\nNo feasible solutions found with current parameters!")
            print("The transport capacity is insufficient for the demand.")
            return [], []
        
        print(f"\nFound {len(solutions)} feasible solutions")
        
        # 转换为数组
        times = np.array([s['construction_time'] for s in solutions])
        costs = np.array([s['total_cost']/1e9 for s in solutions])
        impacts = np.array([s['environmental_impact']/1e3 for s in solutions])
        ratios = np.array([s['elevator_ratio'] for s in solutions])
        
        print(f"\nSTATISTICAL SUMMARY:")
        print(f"Construction time: {min(times)} - {max(times)} years")
        print(f"Total cost: ${min(costs):.2f}B - ${max(costs):.2f}B")
        print(f"Environmental impact: {min(impacts):.1f}k - {max(impacts):.1f}k tons")
        print(f"Elevator ratio: {min(ratios):.1f}% - {max(ratios):.1f}%")
        
        # 找到帕累托前沿
        pareto_front = self.find_pareto_front(solutions)
        print(f"\nPareto front solutions: {len(pareto_front)}")
        
        if pareto_front:
            # 找到极端点
            pareto_times = np.array([s['construction_time'] for s in pareto_front])
            pareto_costs = np.array([s['total_cost']/1e9 for s in pareto_front])
            pareto_impacts = np.array([s['environmental_impact']/1e3 for s in pareto_front])
            
            idx_min_cost = np.argmin(pareto_costs)
            idx_min_time = np.argmin(pareto_times)
            idx_min_impact = np.argmin(pareto_impacts)
            
            # 计算折中解
            norm_costs = pareto_costs / np.max(pareto_costs)
            norm_times = pareto_times / np.max(pareto_times)
            norm_impacts = pareto_impacts / np.max(pareto_impacts)
            distances = np.sqrt(norm_costs**2 + norm_times**2 + norm_impacts**2)
            idx_compromise = np.argmin(distances)
            
            solutions_to_report = [
                ('MINIMUM COST', pareto_front[idx_min_cost]),
                ('MINIMUM TIME', pareto_front[idx_min_time]),
                ('MINIMUM IMPACT', pareto_front[idx_min_impact]),
                ('COMPROMISE', pareto_front[idx_compromise])
            ]
            
            print("\n" + "-"*60)
            print("OPTIMAL SOLUTIONS:")
            print("-"*60)
            
            for title, sol in solutions_to_report:
                print(f"\n{title}:")
                print(f"  Construction time: {sol['construction_time']} years")
                print(f"  Total cost: ${sol['total_cost']/1e9:.2f} billion")
                print(f"  Environmental impact: {sol['environmental_impact']/1e3:.1f} ktons")
                print(f"  Elevator transport: {sol['elevator_transport']/1e6:.2f} million tons")
                print(f"  Rocket transport: {sol['rocket_transport']/1e6:.2f} million tons")
                print(f"  Elevator ratio: {sol['elevator_ratio']:.1f}%")
                print(f"  Annual rocket launches: {sol['annual_rocket_launches']:.0f}")
                print(f"  Total rocket launches: {int(sol['total_rocket_launches']):,}")
        
        return solutions, pareto_front
    
    def create_analysis_plots(self, solutions, pareto_front):
        """创建分析图表"""
        if not solutions:
            return
        
        print("\n" + "="*60)
        print("CREATING ANALYSIS VISUALIZATIONS")
        print("="*60)
        
        # 准备数据
        times = np.array([s['construction_time'] for s in solutions])
        costs = np.array([s['total_cost']/1e9 for s in solutions])
        impacts = np.array([s['environmental_impact']/1e3 for s in solutions])
        ratios = np.array([s['elevator_ratio'] for s in solutions])
        annual_demands = np.array([s['annual_demand']/1e3 for s in solutions])  # ktons
        
        if pareto_front:
            pareto_times = np.array([s['construction_time'] for s in pareto_front])
            pareto_costs = np.array([s['total_cost']/1e9 for s in pareto_front])
            pareto_impacts = np.array([s['environmental_impact']/1e3 for s in pareto_front])
            pareto_ratios = np.array([s['elevator_ratio'] for s in pareto_front])
        
        # 图表1: 容量需求分析
        print("\n1. Capacity and Demand Analysis")
        fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 年需求 vs 容量
        ax1 = axes[0, 0]
        ax1.plot(times, annual_demands, 'bo-', label='Annual Demand', linewidth=2)
        ax1.axhline(y=self.max_total_transport_per_year/1e3, color='r', linestyle='--', 
                   label=f'Max Capacity: {self.max_total_transport_per_year/1e3:.1f} ktons/year')
        ax1.axhline(y=self.elevator_capacity_per_year/1e3, color='g', linestyle='--',
                   label=f'Elevator Capacity: {self.elevator_capacity_per_year/1e3:.1f} ktons/year')
        ax1.axhline(y=self.max_rocket_transport_per_year/1e3, color='orange', linestyle='--',
                   label=f'Rocket Capacity: {self.max_rocket_transport_per_year/1e3:.1f} ktons/year')
        
        ax1.set_xlabel('Construction Time (Years)', fontsize=11)
        ax1.set_ylabel('Annual Transport (k tons)', fontsize=11)
        ax1.set_title('Annual Demand vs Capacity Limits', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # 成本 vs 时间
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(times, costs, c=impacts, cmap='plasma', s=50, alpha=0.7)
        if pareto_front:
            ax2.scatter(pareto_times, pareto_costs, c='red', s=100, marker='*', 
                       label='Pareto Front', edgecolors='black')
        
        ax2.set_xlabel('Construction Time (Years)', fontsize=11)
        ax2.set_ylabel('Total Cost (Billion USD)', fontsize=11)
        ax2.set_title('Cost vs Construction Time', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Environmental Impact (k tons)')
        if pareto_front:
            ax2.legend()
        
        # 环境影响 vs 时间
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(times, impacts, c=costs, cmap='viridis', s=50, alpha=0.7)
        if pareto_front:
            ax3.scatter(pareto_times, pareto_impacts, c='red', s=100, marker='*', 
                       label='Pareto Front', edgecolors='black')
        
        ax3.set_xlabel('Construction Time (Years)', fontsize=11)
        ax3.set_ylabel('Environmental Impact (k tons)', fontsize=11)
        ax3.set_title('Environmental Impact vs Time', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='Total Cost (Billion USD)')
        if pareto_front:
            ax3.legend()
        
        # 电梯比例 vs 时间
        ax4 = axes[1, 1]
        scatter4 = ax4.scatter(times, ratios, c=costs, cmap='coolwarm', s=50, alpha=0.7)
        if pareto_front:
            ax4.scatter(pareto_times, pareto_ratios, c='red', s=100, marker='*', 
                       label='Pareto Front', edgecolors='black')
        
        ax4.set_xlabel('Construction Time (Years)', fontsize=11)
        ax4.set_ylabel('Elevator Transport Ratio (%)', fontsize=11)
        ax4.set_title('Elevator Ratio vs Construction Time', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Total Cost (Billion USD)')
        if pareto_front:
            ax4.legend()
        
        plt.suptitle('Lunar Transport System Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('lunar_transport_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 图表2: 帕累托前沿详细分析
        if pareto_front and len(pareto_front) >= 3:
            print("\n2. Pareto Front Detailed Analysis")
            
            # 找到代表性解决方案
            idx_min_cost = np.argmin(pareto_costs)
            idx_min_time = np.argmin(pareto_times)
            idx_min_impact = np.argmin(pareto_impacts)
            
            norm_costs = pareto_costs / np.max(pareto_costs)
            norm_times = pareto_times / np.max(pareto_times)
            norm_impacts = pareto_impacts / np.max(pareto_impacts)
            distances = np.sqrt(norm_costs**2 + norm_times**2 + norm_impacts**2)
            idx_compromise = np.argmin(distances)
            
            solutions_to_plot = [
                ('Min Cost', pareto_front[idx_min_cost]),
                ('Min Time', pareto_front[idx_min_time]),
                ('Min Impact', pareto_front[idx_min_impact]),
                ('Compromise', pareto_front[idx_compromise])
            ]
            
            fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            # 目标值对比
            ax1 = axes[0, 0]
            categories = ['Cost (B$)', 'Time (Years)', 'Impact (k tons)']
            x = np.arange(len(categories))
            width = 0.2
            
            for i, (label, sol) in enumerate(solutions_to_plot):
                values = [sol['total_cost']/1e9, sol['construction_time'], sol['environmental_impact']/1e3]
                ax1.bar(x + i*width - 1.5*width, values, width, label=label, color=colors[i], alpha=0.7)
            
            ax1.set_xlabel('Objective', fontsize=11)
            ax1.set_ylabel('Value', fontsize=11)
            ax1.set_title('Objective Values Comparison', fontsize=13, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 运输方式对比
            ax2 = axes[0, 1]
            labels = [label for label, _ in solutions_to_plot]
            elevator_values = [sol['elevator_ratio'] for _, sol in solutions_to_plot]
            rocket_values = [100 - ratio for ratio in elevator_values]
            
            bar_width = 0.6
            x = np.arange(len(labels))
            
            bars1 = ax2.bar(x, elevator_values, bar_width, label='Elevator', color='green', alpha=0.7)
            bars2 = ax2.bar(x, rocket_values, bar_width, bottom=elevator_values, label='Rocket', color='orange', alpha=0.7)
            
            ax2.set_xlabel('Solution Type', fontsize=11)
            ax2.set_ylabel('Transport Ratio (%)', fontsize=11)
            ax2.set_title('Transport Mode Comparison', fontsize=13, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 成本构成
            ax3 = axes[0, 2]
            for i, (label, sol) in enumerate(solutions_to_plot):
                elevator_cost = sol['elevator_transport'] * self.elevator_cost_per_ton / 1e9
                rocket_cost = sol['total_rocket_launches'] * self.rocket_cost_per_launch / 1e9
                total_cost = elevator_cost + rocket_cost
                
                ax3.bar(i, elevator_cost, color='green', alpha=0.7, label='Elevator' if i==0 else '')
                ax3.bar(i, rocket_cost, bottom=elevator_cost, color='orange', alpha=0.7, label='Rocket' if i==0 else '')
                
                ax3.text(i, total_cost/2, f'${total_cost:.1f}B', ha='center', va='center', 
                        fontsize=9, fontweight='bold', color='white')
            
            ax3.set_xlabel('Solution Type', fontsize=11)
            ax3.set_ylabel('Cost (Billion USD)', fontsize=11)
            ax3.set_title('Cost Composition', fontsize=13, fontweight='bold')
            ax3.set_xticks(range(len(labels)))
            ax3.set_xticklabels(labels, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 3D帕累托前沿
            ax4 = fig2.add_subplot(2, 3, 4, projection='3d')
            scatter_all = ax4.scatter(times, costs, impacts, c=ratios, cmap='rainbow', s=30, alpha=0.6)
            ax4.scatter(pareto_times, pareto_costs, pareto_impacts, c='red', s=100, 
                       marker='*', label='Pareto Front', edgecolors='black')
            
            ax4.set_xlabel('Construction Time (Years)', fontsize=10, labelpad=10)
            ax4.set_ylabel('Total Cost (B$)', fontsize=10, labelpad=10)
            ax4.set_zlabel('Environmental Impact (k tons)', fontsize=10, labelpad=10)
            ax4.set_title('3D Pareto Frontier', fontsize=13, fontweight='bold')
            fig2.colorbar(scatter_all, ax=ax4, shrink=0.7, label='Elevator Ratio (%)')
            ax4.legend()
            ax4.view_init(elev=25, azim=45)
            
            # 年度运输计划
            ax5 = axes[1, 1]
            solution_idx = idx_compromise
            comp_sol = pareto_front[solution_idx]
            
            years = np.arange(1, comp_sol['construction_time'] + 1)
            elevator_annual = np.full(len(years), comp_sol['annual_elevator']/1e3)
            rocket_annual = np.full(len(years), comp_sol['annual_rocket']/1e3)
            
            bar_width = 0.35
            x = np.arange(len(years))
            
            ax5.bar(x - bar_width/2, elevator_annual, bar_width, label='Elevator', color='green', alpha=0.7)
            ax5.bar(x + bar_width/2, rocket_annual, bar_width, label='Rocket', color='orange', alpha=0.7)
            
            ax5.set_xlabel('Year', fontsize=11)
            ax5.set_ylabel('Annual Transport (k tons)', fontsize=11)
            ax5.set_title(f'Annual Transport Plan (Compromise Solution)', fontsize=13, fontweight='bold')
            ax5.set_xticks(x[::max(1, len(years)//10)])
            ax5.set_xticklabels(years[::max(1, len(years)//10)].astype(int))
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
            
            # 效率雷达图
            ax6 = fig2.add_subplot(2, 3, 6, polar=True)
            
            # 归一化数据
            def normalize_data(data_list):
                data_array = np.array(data_list)
                if np.max(data_array) - np.min(data_array) > 0:
                    return (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))
                else:
                    return np.ones_like(data_array) * 0.5
            
            # 准备雷达图数据
            radar_metrics = ['Cost Efficiency', 'Time Efficiency', 'Env. Efficiency', 'Elevator Usage']
            
            radar_data = []
            for _, sol in solutions_to_plot:
                # 注意：值越小越好，所以效率 = 1/值
                cost_eff = 1 / (sol['total_cost']/1e9) if sol['total_cost'] > 0 else 0
                time_eff = 1 / sol['construction_time'] if sol['construction_time'] > 0 else 0
                env_eff = 1 / (sol['environmental_impact']/1e3) if sol['environmental_impact'] > 0 else 0
                elevator_usage = sol['elevator_ratio'] / 100
                
                efficiency_vector = [cost_eff*1e9, time_eff*10, env_eff*100, elevator_usage]
                radar_data.append(normalize_data(efficiency_vector))
            
            # 绘制雷达图
            angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            for i, (data, (label, _)) in enumerate(zip(radar_data, solutions_to_plot)):
                data = np.concatenate((data, [data[0]]))
                ax6.plot(angles, data, 'o-', linewidth=2, label=label, color=colors[i])
                ax6.fill(angles, data, alpha=0.1, color=colors[i])
            
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(radar_metrics, fontsize=10)
            ax6.set_ylim(0, 1)
            ax6.set_title('Efficiency Comparison (Radar Chart)', fontsize=13, fontweight='bold')
            ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax6.grid(True)
            
            plt.suptitle('Pareto Front Solutions Detailed Analysis', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig('pareto_detailed_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("\n" + "="*60)
        print("VISUALIZATIONS COMPLETED")
        print("="*60)
    
    def generate_recommendations(self, solutions, pareto_front):
        """生成建议"""
        if not solutions or not pareto_front:
            return
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS AND CONCLUSIONS")
        print("="*60)
        
        # 找到代表性解决方案
        pareto_costs = np.array([s['total_cost']/1e9 for s in pareto_front])
        pareto_times = np.array([s['construction_time'] for s in pareto_front])
        pareto_impacts = np.array([s['environmental_impact']/1e3 for s in pareto_front])
        
        idx_min_cost = np.argmin(pareto_costs)
        idx_min_time = np.argmin(pareto_times)
        idx_min_impact = np.argmin(pareto_impacts)
        
        min_cost_sol = pareto_front[idx_min_cost]
        min_time_sol = pareto_front[idx_min_time]
        min_impact_sol = pareto_front[idx_min_impact]
        
        print(f"\nKEY FINDINGS:")
        print(f"1. TRANSPORT CAPACITY:")
        print(f"   • Total annual capacity: {self.max_total_transport_per_year/1e3:.1f} ktons")
        print(f"   • Space elevator contributes: {self.elevator_capacity_per_year/1e3:.1f} ktons ({self.elevator_capacity_per_year/self.max_total_transport_per_year*100:.1f}%)")
        print(f"   • Rockets contribute: {self.max_rocket_transport_per_year/1e3:.1f} ktons ({self.max_rocket_transport_per_year/self.max_total_transport_per_year*100:.1f}%)")
        
        print(f"\n2. OPTIMAL SOLUTIONS:")
        print(f"   A. Minimum Cost (${min_cost_sol['total_cost']/1e9:.2f}B):")
        print(f"      • Time: {min_cost_sol['construction_time']} years")
        print(f"      • Strategy: Maximize cost efficiency")
        
        print(f"\n   B. Minimum Time ({min_time_sol['construction_time']} years):")
        print(f"      • Cost: ${min_time_sol['total_cost']/1e9:.2f}B")
        print(f"      • Strategy: Use maximum capacity")
        
        print(f"\n   C. Minimum Impact ({min_impact_sol['environmental_impact']/1e3:.1f}k tons):")
        print(f"      • Time: {min_impact_sol['construction_time']} years")
        print(f"      • Strategy: Maximize elevator usage")
        
        print(f"\n3. ENVIRONMENTAL IMPACT:")
        print(f"   • Space elevator: Zero emissions")
        print(f"   • Rocket emissions: {self.rocket_pollution_per_launch/self.rocket_payload:.1f} tons/ton")
        print(f"   • Total emissions range: {min(pareto_impacts):.0f}k - {max(pareto_impacts):.0f}k tons")
        
        print(f"\n4. COST ANALYSIS:")
        print(f"   • Space elevator cost: ${self.elevator_cost_per_ton}/ton")
        print(f"   • Rocket cost: ${self.rocket_cost_per_launch/self.rocket_payload:.0f}/ton")
        print(f"   • Cost ratio (Rocket/Elevator): {self.rocket_cost_per_launch/self.rocket_payload/self.elevator_cost_per_ton:.1f}x")
        
        print(f"\n5. RECOMMENDATIONS:")
        print(f"   • For sustainability: Maximize space elevator usage")
        print(f"   • For cost efficiency: Balance between elevator and rockets")
        print(f"   • For speed: Use maximum capacity of both systems")
        print(f"   • Overall: Hybrid system provides best trade-off")
        
        print(f"\n" + "="*60)
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("="*60)
        print("RUNNING REALISTIC LUNAR TRANSPORT ANALYSIS")
        print("="*60)
        
        start_time = time.time()
        
        # 1. 计算可行解
        print("\n1. Calculating feasible solutions...")
        solutions = self.calculate_feasible_solutions()
        
        # 2. 分析
        print("\n2. Analyzing solutions...")
        solutions, pareto_front = self.analyze_solutions(solutions)
        
        if not solutions:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE - NO FEASIBLE SOLUTIONS")
            print("="*60)
            print(f"\nAnalysis time: {time.time() - start_time:.2f} seconds")
            return
        
        # 3. 可视化
        print("\n3. Creating visualizations...")
        self.create_analysis_plots(solutions, pareto_front)
        
        # 4. 建议
        print("\n4. Generating recommendations...")
        self.generate_recommendations(solutions, pareto_front)
        
        end_time = time.time()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        print(f"\nSummary:")
        print(f"  Analysis time: {end_time - start_time:.2f} seconds")
        print(f"  Feasible solutions: {len(solutions)}")
        print(f"  Pareto front solutions: {len(pareto_front)}")
        print(f"  Visualization files created: 2 PNG files")
        
        # 保存数据
        if solutions:
            df = pd.DataFrame(solutions)
            df.to_csv('realistic_lunar_transport_solutions.csv', index=False)
            print(f"\nData saved to: realistic_lunar_transport_solutions.csv")
        
        print("\n" + "="*60)

def main():
    """主函数"""
    print("="*60)
    print("REALISTIC LUNAR TRANSPORT MODEL")
    print("="*60)
    print("Model for 100 million tons of construction materials")
    print("Three-Objective Optimization:")
    print("  1. Minimize Total Cost")
    print("  2. Minimize Construction Time")
    print("  3. Minimize Environmental Impact")
    print("="*60)
    
    # 创建模型
    model = RealisticLunarTransportModel()
    
    # 运行分析
    model.run_complete_analysis()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()