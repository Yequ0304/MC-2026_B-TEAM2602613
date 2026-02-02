# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 14:23:29 2026

@author: admin
"""

"""
月球殖民地水资源供给优化模型
模型V：分阶段混合稳健供给模型 - 单个图表展示版
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# 颜色方案
COLOR_SCHEME = [
    '#6AD1A3',  # 绿色
    '#7FBDDA',  # 蓝色
    '#BBC7BE',  # 灰绿色
    '#FFD47D',  # 黄色
    '#FFA288',  # 橙色
    '#C49892',  # 棕色
    '#929EAB',  # 灰色
    '#84ADDC'   # 浅蓝色
]

class WaterSupplyGuaranteedModelSinglePlots:
    """保证供水的水资源供给优化模型 - 单个图表展示版"""
    
    def __init__(self):
        # 基本参数
        self.N_pop = 100000  # 人口: 10万人
        self.omega_per = 5  # 人均年用水量: 5吨/人/年
        self.T_op = 30  # 运营阶段: 30年
        
        # 水循环参数
        self.eta_base = 0.05  # 初始循环效率
        self.eta_max = 0.98  # 最大循环效率
        self.k = 0.2  # 效率提升速率参数
        self.t0 = 10  # 效率提升拐点年
        
        # 安全储备参数
        self.f_safe = 0.1  # 安全储备系数
        self.W_emergency = self.N_pop * self.omega_per * 30/365  # 30天应急用水储备
        
        # 运输成本参数
        self.c_elevator = 500000  # 太空电梯单位成本: 500,000美元/吨
        self.c_rocket = 1e6  # 火箭单次发射成本: 1百万美元
        
        # 太空电梯参数
        self.C_ideal = 179000  # 理想年运输能力: 17.9万吨/年
        self.alpha = 0.95  # 容量衰减系数
        self.pf = 0.01  # 太空电梯故障概率
        self.num_ports = 3  # 银河港数量
        
        # 火箭参数
        self.launch_sites = [
            {"name": "French Guiana", "lat": 5.2, "category": "low", "payload": 145, "si": 1.0},
            {"name": "Satish Dhawan", "lat": 13.7, "category": "low", "payload": 145, "si": 1.0},
            {"name": "California", "lat": 34.5, "category": "medium", "payload": 125, "si": 0.9},
            {"name": "Texas", "lat": 29.4, "category": "medium", "payload": 125, "si": 0.9},
            {"name": "Florida", "lat": 28.4, "category": "medium", "payload": 125, "si": 0.9},
            {"name": "Virginia", "lat": 37.5, "category": "medium", "payload": 125, "si": 0.9},
            {"name": "Taiyuan", "lat": 38.5, "category": "medium", "payload": 125, "si": 0.9},
            {"name": "Mahia", "lat": 39.0, "category": "medium", "payload": 125, "si": 0.9},
            {"name": "Alaska", "lat": 64.0, "category": "high", "payload": 105, "si": 0.8},
            {"name": "Kazakhstan", "lat": 45.0, "category": "high", "payload": 105, "si": 0.8}
        ]
        self.Y_max = 50  # 每个发射场年发射上限
        self.ps = 0.98  # 火箭发射成功率
        self.pd = 0.10  # 火箭发射延迟率
        self.pp = 0.02  # 火箭部分失败概率
        self.delta = 0.3  # 部分失败载荷损失比例
        
        # 安全系数
        self.safety_factor = 1.3  # 增加30%的安全余量
        
        # 初始化随机种子
        np.random.seed(42)
        
        # 预计算参数
        self.eta_t = self.calculate_efficiency_over_time()
        self.W_supply_t = self.calculate_water_demand()
        
        print("="*60)
        print("WATER SUPPLY GUARANTEED MODEL WITH INDIVIDUAL PLOTS")
        print("="*60)
        print(f"Population: {self.N_pop:,}")
        print(f"Annual water demand per person: {self.omega_per} tons")
        print(f"Total 30-year demand: {np.sum(self.W_supply_t)/1e6:.2f} million tons")
        print(f"Emergency water reserve: {self.W_emergency/1e3:.1f} thousand tons")
        print(f"Safety factor: {self.safety_factor}")
        print(f"Number of launch sites: {len(self.launch_sites)}")
        print("="*60)
        
    def calculate_efficiency_over_time(self):
        """计算随时间变化的水循环效率"""
        eta_t = np.zeros(self.T_op)
        for t in range(self.T_op):
            year = t + 1
            eta_t[t] = self.eta_base + (self.eta_max - self.eta_base) / (1 + np.exp(-self.k * (year - self.t0)))
        return eta_t
    
    def calculate_water_demand(self):
        """计算外部供水需求"""
        W_supply = np.zeros(self.T_op)
        for t in range(self.T_op):
            W_supply[t] = self.N_pop * self.omega_per * (1 - self.eta_t[t]) * (1 + self.f_safe)
        return W_supply
    
    def calculate_transport_capacity(self):
        """计算运输系统的理论容量"""
        n_sites = len(self.launch_sites)
        
        # 1. 太空电梯容量计算
        # 理想情况：3个端口都正常
        max_elevator_per_year = self.C_ideal * self.num_ports * self.alpha
        # 考虑故障概率
        avg_elevator_ports = self.num_ports * (1 - self.pf)
        avg_elevator_per_year = self.C_ideal * avg_elevator_ports * self.alpha
        
        # 2. 火箭容量计算
        max_rocket_per_year = 0
        avg_rocket_per_year = 0
        
        for site in self.launch_sites:
            # 最大容量
            max_payload = site['payload'] * self.Y_max
            max_rocket_per_year += max_payload
            
            # 平均容量（考虑成功率、延迟率、部分失败）
            effective_launches = self.Y_max * self.ps * (1 - self.pd)
            effective_payload = effective_launches * site['payload'] * (1 - self.pp + self.pp * (1 - self.delta))
            avg_rocket_per_year += effective_payload
        
        # 3. 总容量
        max_total_per_year = max_elevator_per_year + max_rocket_per_year
        avg_total_per_year = avg_elevator_per_year + avg_rocket_per_year
        
        return {
            'max_elevator': max_elevator_per_year,
            'avg_elevator': avg_elevator_per_year,
            'max_rocket': max_rocket_per_year,
            'avg_rocket': avg_rocket_per_year,
            'max_total': max_total_per_year,
            'avg_total': avg_total_per_year
        }
    
    def design_guaranteed_solution(self):
        """设计保证供水的解决方案"""
        n_sites = len(self.launch_sites)
        capacity = self.calculate_transport_capacity()
        
        print("\n" + "="*60)
        print("TRANSPORT CAPACITY ANALYSIS")
        print("="*60)
        print(f"Maximum elevator capacity: {capacity['max_elevator']/1e3:.1f} thousand tons/year")
        print(f"Average elevator capacity: {capacity['avg_elevator']/1e3:.1f} thousand tons/year")
        print(f"Maximum rocket capacity: {capacity['max_rocket']/1e3:.1f} thousand tons/year")
        print(f"Average rocket capacity: {capacity['avg_rocket']/1e3:.1f} thousand tons/year")
        print(f"Maximum total capacity: {capacity['max_total']/1e3:.1f} thousand tons/year")
        print(f"Average total capacity: {capacity['avg_total']/1e3:.1f} thousand tons/year")
        
        # 计算年度需求
        max_demand = np.max(self.W_supply_t)
        avg_demand = np.mean(self.W_supply_t)
        
        print(f"\nMaximum annual demand: {max_demand/1e3:.1f} thousand tons")
        print(f"Average annual demand: {avg_demand/1e3:.1f} thousand tons")
        
        # 检查容量是否足够
        if capacity['avg_total'] < max_demand * self.safety_factor:
            print(f"\n⚠ WARNING: Average transport capacity ({capacity['avg_total']/1e3:.1f} ktons/year)")
            print(f"  is less than maximum demand with safety factor ({max_demand*self.safety_factor/1e3:.1f} ktons/year)")
            print("  Increasing rocket launches or considering alternative solutions...")
        
        # 设计运输计划
        x_t = np.zeros(self.T_op)  # 太空电梯
        y_i_t = np.zeros((n_sites, self.T_op))  # 火箭
        
        # 策略：优先使用太空电梯，不足部分用火箭补充
        for t in range(self.T_op):
            demand = self.W_supply_t[t] * self.safety_factor  # 考虑安全系数
            
            # 太空电梯承担部分
            # 太空电梯承担60-80%的需求
            elevator_share = 0.7 + 0.1 * np.sin(t/5)  # 随时间变化
            elevator_target = demand * elevator_share
            
            # 确保不超过平均容量
            x_t[t] = min(elevator_target, capacity['avg_elevator'])
            
            # 火箭承担剩余部分
            rocket_needed = demand - x_t[t]
            
            if rocket_needed > 0:
                # 在发射场之间分配
                total_efficiency = sum(site['si'] for site in self.launch_sites)
                
                for i, site in enumerate(self.launch_sites):
                    # 按效率因子分配
                    share = site['si'] / total_efficiency
                    site_need = rocket_needed * share
                    
                    # 计算需要的发射次数
                    launches_needed = site_need / site['payload']
                    
                    # 考虑不确定性，增加余量
                    launches_needed *= 1.2
                    
                    # 确保不超过年发射上限
                    y_i_t[i, t] = min(launches_needed, self.Y_max)
        
        return np.concatenate([x_t, y_i_t.flatten()])
    
    def evaluate_solution(self, solution, n_scenarios=50):
        """评估解决方案的可靠性"""
        n_sites = len(self.launch_sites)
        x_t = solution[:self.T_op]
        y_i_t = solution[self.T_op:].reshape(n_sites, self.T_op)
        
        total_costs = []
        reliabilities = []
        shortage_details = []
        
        for s in range(n_scenarios):
            cost = 0
            inventory = self.W_emergency
            shortage_years = 0
            annual_shortages = []
            
            for t in range(self.T_op):
                year_shortage = 0
                
                # 太空电梯运输（考虑故障）
                ports_working = np.sum(np.random.binomial(1, 1 - self.pf, self.num_ports))
                elevator_capacity = self.alpha * self.C_ideal * ports_working
                elevator_transport = min(x_t[t], elevator_capacity)
                cost += elevator_transport * self.c_elevator
                
                # 火箭运输（考虑不确定性）
                rocket_transport = 0
                for i in range(n_sites):
                    # 模拟每次发射
                    site_launches = int(np.round(y_i_t[i, t]))
                    site_payloads = []
                    
                    for _ in range(site_launches):
                        # 是否成功
                        is_success = np.random.binomial(1, self.ps)
                        if is_success:
                            # 是否部分失败
                            is_partial = np.random.binomial(1, self.pp)
                            payload = self.launch_sites[i]['payload']
                            if is_partial:
                                payload *= (1 - self.delta)
                        else:
                            payload = 0
                        site_payloads.append(payload)
                    
                    # 考虑延迟
                    is_delayed = np.random.binomial(1, self.pd)
                    if is_delayed:
                        effective_payload = np.sum(site_payloads) * 0.9
                    else:
                        effective_payload = np.sum(site_payloads)
                    
                    rocket_transport += effective_payload
                    cost += site_launches * self.c_rocket
                
                # 更新库存
                total_transport = elevator_transport + rocket_transport
                inventory = inventory + total_transport - self.W_supply_t[t]
                
                # 检查短缺
                if inventory < 0:
                    shortage_years += 1
                    year_shortage = -inventory
                    inventory = 0
                
                annual_shortages.append(year_shortage)
            
            # 计算可靠性
            reliability = 1 - (shortage_years / self.T_op)
            
            total_costs.append(cost)
            reliabilities.append(reliability)
            shortage_details.append(annual_shortages)
        
        # 计算统计量
        expected_cost = np.mean(total_costs)
        expected_reliability = np.mean(reliabilities)
        avg_shortage = np.mean([np.sum(s) for s in shortage_details])
        
        return expected_cost, expected_reliability, avg_shortage
    
    def analyze_solution(self, solution):
        """详细分析解决方案"""
        n_sites = len(self.launch_sites)
        x_t = solution[:self.T_op]
        y_i_t = solution[self.T_op:].reshape(n_sites, self.T_op)
        
        # 转换为整数
        y_i_t_int = np.round(y_i_t).astype(int)
        
        # 计算统计量
        total_elevator = np.sum(x_t)
        total_rocket_launches = np.sum(y_i_t_int)
        avg_launches_per_site = np.mean(np.sum(y_i_t_int, axis=1))
        
        # 计算年度数据
        annual_data = []
        cumulative_supply = 0
        cumulative_demand = 0
        
        for t in range(self.T_op):
            elevator_transport = x_t[t]
            rocket_transport = 0
            for i in range(n_sites):
                rocket_transport += y_i_t_int[i, t] * self.launch_sites[i]['payload']
            
            total_transport = elevator_transport + rocket_transport
            demand = self.W_supply_t[t]
            surplus = total_transport - demand
            
            cumulative_supply += total_transport
            cumulative_demand += demand
            
            annual_data.append({
                'year': t + 1,
                'demand': demand,
                'elevator': elevator_transport,
                'rocket': rocket_transport,
                'total': total_transport,
                'surplus': surplus,
                'efficiency': self.eta_t[t],
                'cumulative_supply': cumulative_supply,
                'cumulative_demand': cumulative_demand
            })
        
        return {
            'total_elevator': total_elevator,
            'total_rocket_launches': total_rocket_launches,
            'avg_launches_per_site': avg_launches_per_site,
            'annual_data': annual_data,
            'x_t': x_t,
            'y_i_t': y_i_t_int
        }
    
    def run_analysis(self):
        """运行完整分析"""
        print("="*60)
        print("RUNNING GUARANTEED WATER SUPPLY ANALYSIS")
        print("="*60)
        
        start_time = time.time()
        
        # 设计保证供水的解决方案
        print("\n1. Designing guaranteed water supply solution...")
        solution = self.design_guaranteed_solution()
        
        # 评估解决方案
        print("\n2. Evaluating solution reliability...")
        cost, reliability, avg_shortage = self.evaluate_solution(solution, n_scenarios=100)
        
        # 分析解决方案
        print("\n3. Analyzing solution details...")
        analysis = self.analyze_solution(solution)
        
        end_time = time.time()
        
        # 生成报告
        self.generate_report(cost, reliability, avg_shortage, analysis, end_time-start_time)
        
        # 单独显示每个图表
        self.visualize_individual_results(solution, analysis)
        
        return solution, cost, reliability, analysis
    
    def generate_report(self, cost, reliability, avg_shortage, analysis, runtime):
        """生成分析报告"""
        report = []
        report.append("="*80)
        report.append("GUARANTEED WATER SUPPLY SOLUTION")
        report.append("="*80)
        report.append("")
        
        # 基本参数
        report.append("MODEL PARAMETERS:")
        report.append("-"*40)
        report.append(f"Population: {self.N_pop:,}")
        report.append(f"Annual water per person: {self.omega_per} tons")
        report.append(f"Operation period: {self.T_op} years")
        report.append(f"Safety factor: {self.safety_factor}")
        report.append(f"Emergency reserve: {self.W_emergency/1e3:.1f} thousand tons")
        report.append("")
        
        # 性能结果
        report.append("PERFORMANCE RESULTS:")
        report.append("-"*40)
        report.append(f"Total cost: ${cost/1e9:.2f} billion")
        report.append(f"System reliability: {reliability*100:.2f}%")
        report.append(f"Average annual shortage: {avg_shortage/1e3:.1f} thousand tons")
        report.append("")
        
        # 运输计划
        report.append("TRANSPORTATION PLAN:")
        report.append("-"*40)
        report.append(f"Total elevator transport: {analysis['total_elevator']/1e6:.2f} million tons")
        report.append(f"Total rocket launches: {analysis['total_rocket_launches']:,}")
        report.append(f"Average launches per site: {analysis['avg_launches_per_site']:.1f}")
        report.append("")
        
        # 累积数据
        final_year = analysis['annual_data'][-1]
        total_supply = final_year['cumulative_supply']
        total_demand = final_year['cumulative_demand']
        total_surplus = total_supply - total_demand
        
        report.append("CUMULATIVE DATA (30 YEARS):")
        report.append("-"*40)
        report.append(f"Total water demand: {total_demand/1e6:.2f} million tons")
        report.append(f"Total water supply: {total_supply/1e6:.2f} million tons")
        report.append(f"Overall surplus/deficit: {total_surplus/1e6:+.3f} million tons")
        
        if total_surplus >= 0:
            report.append("✓ WATER SUPPLY IS GUARANTEED")
        else:
            report.append(f"⚠ WATER DEFICIT: {-total_surplus/1e6:.3f} million tons")
        report.append("")
        
        # 年度数据（前5年）
        report.append("ANNUAL PERFORMANCE (FIRST 5 YEARS):")
        report.append("-"*40)
        for t in range(min(5, self.T_op)):
            data = analysis['annual_data'][t]
            status = "SURPLUS" if data['surplus'] >= 0 else "DEFICIT"
            report.append(f"Year {data['year']} ({status}):")
            report.append(f"  Demand: {data['demand']/1e3:.1f} ktons (Efficiency: {data['efficiency']*100:.1f}%)")
            report.append(f"  Supply: {data['total']/1e3:.1f} ktons (Elevator: {data['elevator']/1e3:.1f} ktons, Rocket: {data['rocket']/1e3:.1f} ktons)")
            report.append(f"  Balance: {data['surplus']/1e3:+.1f} ktons")
        
        if self.T_op > 5:
            report.append(f"... and {self.T_op-5} more years")
        report.append("")
        
        # 发射场使用情况
        report.append("LAUNCH SITE UTILIZATION:")
        report.append("-"*40)
        n_sites = len(self.launch_sites)
        y_i_t = analysis['y_i_t']
        
        for i in range(n_sites):
            total_launches = np.sum(y_i_t[i, :])
            site_name = self.launch_sites[i]['name']
            avg_per_year = total_launches / self.T_op
            utilization = (avg_per_year / self.Y_max) * 100
            
            report.append(f"{site_name}: {int(total_launches):,} launches ({avg_per_year:.1f}/year, {utilization:.1f}% of capacity)")
        report.append("")
        
        # 技术指标
        report.append("TECHNICAL METRICS:")
        report.append("-"*40)
        report.append(f"Analysis runtime: {runtime:.2f} seconds")
        report.append(f"Number of scenarios evaluated: 100")
        report.append("")
        
        # 建议
        report.append("RECOMMENDATIONS:")
        report.append("-"*40)
        
        if reliability >= 0.99:
            report.append("✓ EXCELLENT: Solution provides near-perfect reliability")
        elif reliability >= 0.95:
            report.append("✓ VERY GOOD: Solution provides high reliability")
        elif reliability >= 0.9:
            report.append("✓ GOOD: Solution provides reliable water supply")
        elif reliability >= 0.8:
            report.append("⚠ MODERATE: Acceptable reliability with some risk")
        else:
            report.append("⚠ POOR: Significant reliability issues detected")
        
        if total_surplus >= 0:
            report.append("✓ Cumulative water supply meets or exceeds demand")
        else:
            report.append("⚠ Cumulative water deficit detected - increase transport capacity")
        
        report.append("")
        report.append("ACTIONS:")
        report.append("1. Implement the designed transportation plan")
        report.append("2. Monitor water recycling efficiency continuously")
        report.append("3. Maintain 30-day emergency water reserves")
        report.append("4. Regularly audit transport system performance")
        report.append("5. Consider technology upgrades for higher efficiency")
        
        print("\n" + "\n".join(report))
        
        # 保存报告
        with open("guaranteed_water_supply_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report))
    
    def visualize_individual_results(self, solution, analysis):
        """单独展示每个图表"""
        n_sites = len(self.launch_sites)
        x_t = solution[:self.T_op]
        y_i_t = solution[self.T_op:].reshape(n_sites, self.T_op)
        
        years = np.arange(1, self.T_op + 1)
        
        print("\n" + "="*60)
        print("GENERATING INDIVIDUAL VISUALIZATION CHARTS")
        print("="*60)
        
        # 图表1: 水循环效率
        print("\nChart 1: Water Recycling Efficiency Over Time")
        plt.figure(figsize=(10, 6))
        plt.plot(years, self.eta_t * 100, 'o-', linewidth=2, markersize=4, color=COLOR_SCHEME[0])
        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Recycling Efficiency (%)', fontsize=12, fontweight='bold')
        plt.title('Water Recycling Efficiency Over Time', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(0, 100)
        
        # 标记拐点
        t0_idx = self.t0 - 1
        if 0 <= t0_idx < self.T_op:
            plt.scatter(self.t0, self.eta_t[t0_idx] * 100, s=100, color=COLOR_SCHEME[4], zorder=5)
            plt.annotate(f'Inflection Point\nYear {self.t0}', xy=(self.t0, self.eta_t[t0_idx] * 100),
                        xytext=(self.t0 + 2, self.eta_t[t0_idx] * 100 - 10),
                        arrowprops=dict(arrowstyle='->', color='black'),
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 图表2: 年度供需平衡
        print("\nChart 2: Annual Water Supply vs Demand")
        plt.figure(figsize=(10, 6))
        
        demand = np.array([d['demand'] for d in analysis['annual_data']]) / 1e3
        supply = np.array([d['total'] for d in analysis['annual_data']]) / 1e3
        
        plt.plot(years, demand, 's-', linewidth=2, markersize=4, color=COLOR_SCHEME[1], label='Demand')
        plt.plot(years, supply, 'o-', linewidth=2, markersize=4, color=COLOR_SCHEME[2], label='Supply')
        
        # 填充盈余/赤字区域
        plt.fill_between(years, demand, supply, where=supply >= demand,
                        color=COLOR_SCHEME[0], alpha=0.3, label='Surplus')
        plt.fill_between(years, demand, supply, where=supply < demand,
                        color=COLOR_SCHEME[4], alpha=0.3, label='Deficit')
        
        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Water (thousand tons)', fontsize=12, fontweight='bold')
        plt.title('Annual Water Supply vs Demand', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # 图表3: 累积供需
        print("\nChart 3: Cumulative Water Supply vs Demand")
        plt.figure(figsize=(10, 6))
        
        cum_demand = np.array([d['cumulative_demand'] for d in analysis['annual_data']]) / 1e6
        cum_supply = np.array([d['cumulative_supply'] for d in analysis['annual_data']]) / 1e6
        
        plt.plot(years, cum_demand, 's-', linewidth=2, markersize=4, color=COLOR_SCHEME[1], label='Cumulative Demand')
        plt.plot(years, cum_supply, 'o-', linewidth=2, markersize=4, color=COLOR_SCHEME[2], label='Cumulative Supply')
        
        # 填充累计盈余区域
        plt.fill_between(years, cum_demand, cum_supply, where=cum_supply >= cum_demand,
                        color=COLOR_SCHEME[0], alpha=0.3, label='Cumulative Surplus')
        
        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Water (million tons)', fontsize=12, fontweight='bold')
        plt.title('Cumulative Water Supply vs Demand', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # 图表4: 运输方式占比
        print("\nChart 4: Transportation Mode Ratio")
        plt.figure(figsize=(10, 6))
        
        elevator_transport = np.array([d['elevator'] for d in analysis['annual_data']])
        rocket_transport = np.array([d['rocket'] for d in analysis['annual_data']])
        total_transport = elevator_transport + rocket_transport
        
        elevator_ratio = (elevator_transport / total_transport * 100)
        rocket_ratio = (rocket_transport / total_transport * 100)
        
        bar_width = 0.8
        plt.bar(years, elevator_ratio, bar_width, color=COLOR_SCHEME[0], label='Elevator', alpha=0.7)
        plt.bar(years, rocket_ratio, bar_width, bottom=elevator_ratio, color=COLOR_SCHEME[4], label='Rocket', alpha=0.7)
        
        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Transport Ratio (%)', fontsize=12, fontweight='bold')
        plt.title('Transportation Mode Ratio', fontsize=14, fontweight='bold', pad=20)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # 图表5: 发射场使用情况
        print("\nChart 5: Top Launch Sites Usage")
        plt.figure(figsize=(10, 6))
        
        site_names = [site['name'] for site in self.launch_sites]
        total_launches = np.sum(y_i_t, axis=1)
        
        # 选择前5个最活跃的
        top_n = min(5, n_sites)
        top_indices = np.argsort(total_launches)[-top_n:][::-1]
        
        top_sites = [site_names[i] for i in top_indices]
        top_launches = [total_launches[i] for i in top_indices]
        
        bars = plt.bar(range(top_n), top_launches, color=COLOR_SCHEME[4])
        plt.xlabel('Launch Site', fontsize=12, fontweight='bold')
        plt.ylabel('Total Launches (30 years)', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Launch Sites Utilization', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(range(top_n), top_sites, rotation=45, ha='right', fontsize=10)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        for bar, launch in zip(bars, top_launches):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(launch)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 图表6: 成本分布
        print("\nChart 6: Cost Distribution")
        plt.figure(figsize=(8, 8))
        
        # 估算成本分布
        elevator_cost = analysis['total_elevator'] * self.c_elevator
        rocket_cost = analysis['total_rocket_launches'] * self.c_rocket
        total_cost = elevator_cost + rocket_cost
        
        labels = ['Space Elevator', 'Rocket']
        sizes = [elevator_cost/total_cost*100, rocket_cost/total_cost*100]
        colors = [COLOR_SCHEME[0], COLOR_SCHEME[4]]
        
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 11})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title('Cost Distribution (30 Years)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # 图表7: 年度运输计划（前10年）
        print("\nChart 7: Annual Transportation Plan (Years 1-10)")
        plt.figure(figsize=(10, 6))
        
        years_short = years[:10]
        
        bar_width = 0.35
        x_pos = np.arange(len(years_short))
        
        elevator_annual = np.array([d['elevator'] for d in analysis['annual_data'][:10]]) / 1e3
        rocket_annual = np.array([d['rocket'] for d in analysis['annual_data'][:10]]) / 1e3
        
        bars1 = plt.bar(x_pos - bar_width/2, elevator_annual, bar_width,
                       label='Elevator', color=COLOR_SCHEME[0])
        bars2 = plt.bar(x_pos + bar_width/2, rocket_annual, bar_width,
                       label='Rocket', color=COLOR_SCHEME[4])
        
        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Annual Transport (thousand tons)', fontsize=12, fontweight='bold')
        plt.title('Annual Transportation Plan (Years 1-10)', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(x_pos, years_short)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加总运输量标签
        for i, (e, r) in enumerate(zip(elevator_annual, rocket_annual)):
            total = e + r
            plt.text(i, max(e, r) + 10, f'{total:.0f}', 
                    ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 图表8: 年度水平衡（前10年）
        print("\nChart 8: Annual Water Balance (Years 1-10)")
        plt.figure(figsize=(10, 6))
        
        demand_annual = demand[:10]
        supply_annual = supply[:10]
        
        plt.plot(years_short, demand_annual, 's-', linewidth=2, markersize=6,
                color=COLOR_SCHEME[1], label='Demand')
        plt.plot(years_short, supply_annual, 'o-', linewidth=2, markersize=6,
                color=COLOR_SCHEME[2], label='Supply')
        
        # 填充区域
        plt.fill_between(years_short, demand_annual, supply_annual,
                        where=supply_annual >= demand_annual,
                        color=COLOR_SCHEME[0], alpha=0.3, label='Surplus')
        plt.fill_between(years_short, demand_annual, supply_annual,
                        where=supply_annual < demand_annual,
                        color=COLOR_SCHEME[4], alpha=0.3, label='Deficit')
        
        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Water (thousand tons)', fontsize=12, fontweight='bold')
        plt.title('Annual Water Balance (Years 1-10)', fontsize=14, fontweight='bold', pad=20)
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
        
        # 图表9: 运输容量分析
        print("\nChart 9: Transport Capacity Analysis")
        plt.figure(figsize=(10, 6))
        
        capacity = self.calculate_transport_capacity()
        
        categories = ['Max Elevator', 'Avg Elevator', 'Max Rocket', 'Avg Rocket', 'Max Total', 'Avg Total']
        values = [capacity['max_elevator']/1e3, capacity['avg_elevator']/1e3,
                 capacity['max_rocket']/1e3, capacity['avg_rocket']/1e3,
                 capacity['max_total']/1e3, capacity['avg_total']/1e3]
        
        colors_bar = [COLOR_SCHEME[0], COLOR_SCHEME[0], COLOR_SCHEME[4], 
                     COLOR_SCHEME[4], COLOR_SCHEME[1], COLOR_SCHEME[1]]
        
        bars = plt.bar(categories, values, color=colors_bar, alpha=0.7)
        
        # 添加需求线
        max_demand = np.max(self.W_supply_t) * self.safety_factor / 1e3
        avg_demand = np.mean(self.W_supply_t) * self.safety_factor / 1e3
        
        plt.axhline(y=max_demand, color='red', linestyle='--', linewidth=2, label=f'Max Demand with Safety ({max_demand:.1f} ktons)')
        plt.axhline(y=avg_demand, color='orange', linestyle='--', linewidth=2, label=f'Avg Demand with Safety ({avg_demand:.1f} ktons)')
        
        plt.xlabel('Capacity Type', fontsize=12, fontweight='bold')
        plt.ylabel('Capacity (thousand tons/year)', fontsize=12, fontweight='bold')
        plt.title('Transport Capacity Analysis', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, f'{value:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 图表10: 年度盈余/赤字趋势
        print("\nChart 10: Annual Surplus/Deficit Trend")
        plt.figure(figsize=(10, 6))
        
        surplus_values = np.array([d['surplus'] for d in analysis['annual_data']]) / 1e3
        
        plt.bar(years, surplus_values, color=np.where(surplus_values >= 0, COLOR_SCHEME[0], COLOR_SCHEME[4]), alpha=0.7)
        plt.axhline(y=0, color='black', linewidth=1)
        
        # 添加趋势线
        if len(years) > 1:
            z = np.polyfit(years, surplus_values, 2)
            p = np.poly1d(z)
            plt.plot(years, p(years), 'r--', linewidth=2, label='Trend Line')
        
        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Surplus/Deficit (thousand tons)', fontsize=12, fontweight='bold')
        plt.title('Annual Water Balance Surplus/Deficit', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 添加累计盈余标注
        total_surplus = np.sum(surplus_values)
        plt.text(0.02, 0.98, f'Total 30-year surplus: {total_surplus:.1f} ktons',
                transform=plt.gca().transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*60)
        print("ALL CHARTS HAVE BEEN GENERATED AND DISPLAYED")
        print("="*60)

def main():
    """主函数"""
    print("="*60)
    print("GUARANTEED WATER SUPPLY FOR LUNAR COLONY")
    print("="*60)
    print("Individual Charts Version - Each chart displayed separately")
    print("="*60)
    
    # 创建模型
    model = WaterSupplyGuaranteedModelSinglePlots()
    
    # 运行分析
    result = model.run_analysis()
    
    if result:
        solution, cost, reliability, analysis = result
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        final_year = analysis['annual_data'][-1]
        total_supply = final_year['cumulative_supply']
        total_demand = final_year['cumulative_demand']
        total_surplus = total_supply - total_demand
        
        print(f"\nKEY RESULTS:")
        print(f"  Total 30-year cost: ${cost/1e9:.2f} billion")
        print(f"  System reliability: {reliability*100:.2f}%")
        print(f"  Total water demand: {total_demand/1e6:.2f} million tons")
        print(f"  Total water supply: {total_supply/1e6:.2f} million tons")
        print(f"  Overall surplus/deficit: {total_surplus/1e6:+.3f} million tons")
        print(f"  Total rocket launches: {analysis['total_rocket_launches']:,}")
        
        if total_surplus >= 0 and reliability >= 0.9:
            print(f"\n✓ SUCCESS: Water supply is GUARANTEED with high reliability!")
        elif total_surplus >= 0 and reliability >= 0.8:
            print(f"\n✓ ACCEPTABLE: Water supply meets demand with acceptable reliability")
        else:
            print(f"\n⚠ WARNING: Water supply may be insufficient or unreliable")
            print("  Consider increasing transport capacity or safety margins")
        
        print("="*60)
    else:
        print("\nAnalysis failed!")

if __name__ == "__main__":
    main()