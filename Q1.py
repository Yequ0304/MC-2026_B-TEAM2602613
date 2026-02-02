# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 21:27:20 2026

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# Set English font and style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.style.use('seaborn-v0_8')

# Data
scenarios = ['Pure Space Elevator', 'Pure Rocket', 'Mixed Transport\n(74.1%:25.9%)']
costs = [50.00, 62.99, 53.62]  # Trillion USD
periods = [558.7, 1574.8, 414.1]  # Years
emissions = [0.0, 7874.02, 2070.39]  # 10k tons CO2e

# Convert lists to numpy arrays
costs = np.array(costs)
periods = np.array(periods)
emissions = np.array(emissions)

# Sensitivity analysis data
elevator_ratios = [0.2, 0.4, 0.6, 0.8, 0.741]
mixed_costs = [61.20, 58.40, 55.60, 52.80, 53.62]
mixed_periods = [1280.0, 960.0, 640.0, 446.9, 414.1]
mixed_emissions = [6400.00, 4800.00, 3200.00, 1600.00, 2070.39]

# 1. 3D Comparison Chart (单独显示)
plt.figure(figsize=(12, 8))
ax1 = plt.axes(projection='3d')
colors = ['#2E86AB', '#A23B72', '#F18F01']

x_pos = np.arange(len(scenarios))
y_pos = np.zeros(len(scenarios))
width = 0.6
depth = 0.4

ax1.bar3d(x_pos - width/2, y_pos, np.zeros(len(scenarios)), 
          width, depth, costs, color=colors, alpha=0.8, 
          edgecolor='black', linewidth=0.5, zorder=3)
ax1.bar3d(x_pos - width/2, y_pos + 0.5, np.zeros(len(scenarios)), 
          width, depth, periods, color=colors, alpha=0.8, 
          edgecolor='black', linewidth=0.5, zorder=3)
ax1.bar3d(x_pos - width/2, y_pos + 1, np.zeros(len(scenarios)), 
          width, depth, emissions/100, color=colors, alpha=0.8, 
          edgecolor='black', linewidth=0.5, zorder=3)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(scenarios, rotation=20, ha='right')
ax1.set_yticks([0, 0.5, 1])
ax1.set_yticklabels(['Cost (Trillion USD)', 'Period (Years)', 'Emissions (/100 10k tons)'], rotation=-10, ha='left')
ax1.set_zlabel('Value')
ax1.set_title('3D Comparison of Three Transport Schemes', fontsize=14, fontweight='bold', pad=20)
ax1.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.savefig('1_3d_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 2. Radar Chart (单独显示)
plt.figure(figsize=(10, 8))
ax2 = plt.subplot(111, projection='polar')

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

norm_cost = 1 - normalize_data(costs)
norm_period = 1 - normalize_data(periods)
norm_emission = 1 - normalize_data(emissions)

categories = ['Cost\n(Lower Better)', 'Period\n(Shorter Better)', 'Emissions\n(Lower Better)']
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()

norm_cost = np.append(norm_cost, norm_cost[0])
norm_period = np.append(norm_period, norm_period[0])
norm_emission = np.append(norm_emission, norm_emission[0])
angles += angles[:1]

ax2.plot(angles, norm_cost, 'o-', linewidth=2, label='Pure Space Elevator', color='#2E86AB')
ax2.fill(angles, norm_cost, alpha=0.25, color='#2E86AB')
ax2.plot(angles, norm_period, 'o-', linewidth=2, label='Pure Rocket', color='#A23B72')
ax2.fill(angles, norm_period, alpha=0.25, color='#A23B72')
ax2.plot(angles, norm_emission, 'o-', linewidth=2, label='Mixed Transport', color='#F18F01')
ax2.fill(angles, norm_emission, alpha=0.25, color='#F18F01')

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories)
ax2.set_ylim(0, 1)
ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
ax2.set_title('Scheme Performance Radar Chart\n(Normalized Comparison, Higher Better)', fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('2_radar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 3. Pareto Frontier Chart (单独显示)
plt.figure(figsize=(10, 8))
ax3 = plt.subplot(111)

scatter1 = ax3.scatter(costs[0], periods[0], s=300, c='#2E86AB', marker='^', edgecolors='black', linewidth=2, zorder=5)
scatter2 = ax3.scatter(costs[1], periods[1], s=300, c='#A23B72', marker='s', edgecolors='black', linewidth=2, zorder=5)
scatter3 = ax3.scatter(costs[2], periods[2], s=300, c='#F18F01', marker='o', edgecolors='black', linewidth=2, zorder=5)

for i, scenario in enumerate(scenarios):
    ax3.annotate(scenario, (costs[i], periods[i]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

pareto_points = sorted([(costs[0], periods[0]), (costs[2], periods[2]), (costs[1], periods[1])], key=lambda x: x[0])
pareto_x, pareto_y = zip(*pareto_points)
ax3.plot(pareto_x, pareto_y, '--', color='gray', alpha=0.7, linewidth=2, label='Pareto Frontier')
ax3.fill_between(pareto_x, 0, pareto_y, alpha=0.1, color='green')

ax3.set_xlabel('Total Cost (Trillion USD)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Construction Period (Years)', fontsize=12, fontweight='bold')
ax3.set_title('Cost-Period Pareto Frontier', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig('3_pareto_frontier.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 4. Sensitivity Analysis Line Chart (单独显示)
plt.figure(figsize=(12, 8))
ax4 = plt.subplot(111)

ax4_secondary = ax4.twinx()

line1, = ax4.plot([r*100 for r in elevator_ratios[:-1]], mixed_costs[:-1], 
                  'o-', color='#2E86AB', linewidth=3, markersize=8, label='Total Cost')
ax4.scatter(elevator_ratios[-1]*100, mixed_costs[-1], s=200, 
            color='red', marker='*', edgecolors='black', linewidth=2, zorder=5, label='Optimal Solution')

line2, = ax4_secondary.plot([r*100 for r in elevator_ratios[:-1]], mixed_periods[:-1], 
                            's-', color='#A23B72', linewidth=3, markersize=8, label='Construction Period')
ax4_secondary.scatter(elevator_ratios[-1]*100, mixed_periods[-1], s=200, 
                      color='red', marker='*', edgecolors='black', linewidth=2, zorder=5)

ax4.set_xlabel('Space Elevator Transport Ratio (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Total Cost (Trillion USD)', fontsize=12, fontweight='bold', color='#2E86AB')
ax4_secondary.set_ylabel('Construction Period (Years)', fontsize=12, fontweight='bold', color='#A23B72')

ax4.tick_params(axis='y', labelcolor='#2E86AB')
ax4_secondary.tick_params(axis='y', labelcolor='#A23B72')

ax4.grid(True, alpha=0.3)
ax4.set_title('Sensitivity Analysis: Transport Ratio vs Cost & Period', fontsize=14, fontweight='bold')

lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax4.legend(lines, labels, loc='upper right')

plt.tight_layout()
plt.savefig('4_sensitivity_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 5. Cost-Benefit Comparison Chart (单独显示)
plt.figure(figsize=(12, 8))
ax5 = plt.subplot(111)

cost_reduction = [(costs[0]-costs[0])/costs[0]*100,
                  (costs[1]-costs[2])/costs[1]*100,
                  (costs[0]-costs[2])/costs[0]*100]

time_reduction = [0,
                  (periods[1]-periods[2])/periods[1]*100,
                  (periods[0]-periods[2])/periods[0]*100]

emission_reduction = [0,
                      (emissions[1]-emissions[2])/emissions[1]*100,
                      (emissions[0]-emissions[2])/emissions[0]*100 if emissions[0] != 0 else 0]

x = np.arange(3)
width = 0.25

bars1 = ax5.bar(x - width, cost_reduction, width, label='Cost Saving', color='#4ECDC4', edgecolor='black', linewidth=1)
bars2 = ax5.bar(x, time_reduction, width, label='Period Reduction', color='#FF6B6B', edgecolor='black', linewidth=1)
bars3 = ax5.bar(x + width, emission_reduction, width, label='Emission Reduction', color='#45B7D1', edgecolor='black', linewidth=1)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax5.set_xlabel('Comparison Scheme', fontsize=12, fontweight='bold')
ax5.set_ylabel('Improvement Percentage (%)', fontsize=12, fontweight='bold')
ax5.set_title('Mixed Scheme Benefits vs Traditional Schemes', fontsize=14, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(['Pure Elevator\n(Baseline)', 'Pure Rocket\nvs Mixed', 'Pure Elevator\nvs Mixed'])
ax5.legend(loc='upper left')
ax5.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('5_cost_benefit.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 6. Stacked Area Chart (单独显示)
plt.figure(figsize=(12, 8))
ax6 = plt.subplot(111)

years = np.arange(1, int(periods[2]) + 1)
elevator_yearly = 179000
rocket_yearly = 2070.39 * 10000 / periods[2]

elevator_cumulative = np.cumsum([elevator_yearly] * len(years))
rocket_cumulative = np.cumsum([rocket_yearly] * len(years))

elevator_cumulative = np.minimum(elevator_cumulative, costs[0] * 1e12 / 500000)
rocket_cumulative = np.minimum(rocket_cumulative, 100000000 - elevator_cumulative)

ax6.fill_between(years, 0, elevator_cumulative/1e6, alpha=0.7, color='#F18F01', label='Space Elevator Transport')
ax6.fill_between(years, elevator_cumulative/1e6, (elevator_cumulative + rocket_cumulative)/1e6, 
                 alpha=0.7, color='#2E86AB', label='Rocket Transport')

ax6.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Total Demand (100M tons)')
ax6.axvline(x=periods[2], color='green', linestyle='--', linewidth=2, label=f'Construction Complete ({int(periods[2])} years)')

ax6.set_xlabel('Construction Year', fontsize=12, fontweight='bold')
ax6.set_ylabel('Cumulative Transport Volume (Million tons)', fontsize=12, fontweight='bold')
ax6.set_title('Mixed Transport Scheme Annual Progress', fontsize=14, fontweight='bold')
ax6.legend(loc='lower right')
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 500)
ax6.set_ylim(0, 120)

plt.tight_layout()
plt.savefig('6_stacked_area.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 7. Final Comparison Chart (单独显示)
plt.figure(figsize=(16, 10))
ax = plt.subplot(111)

x = np.arange(len(scenarios))
width = 0.25

ax1_primary = ax
ax2_secondary = ax.twinx()
ax3_secondary = ax.twinx()
ax3_secondary.spines['right'].set_position(('outward', 60))

bars1 = ax1_primary.bar(x - width, costs, width, label='Total Cost', color='#2E86AB', edgecolor='black', linewidth=1.5)
bars2 = ax2_secondary.bar(x, periods, width, label='Construction Period', color='#A23B72', edgecolor='black', linewidth=1.5)
bars3 = ax3_secondary.bar(x + width, emissions, width, label='Annual Emissions', color='#F18F01', edgecolor='black', linewidth=1.5)

ax1_primary.set_xlabel('Transport Scheme', fontsize=13, fontweight='bold')
ax1_primary.set_ylabel('Total Cost (Trillion USD)', fontsize=12, fontweight='bold', color='#2E86AB')
ax2_secondary.set_ylabel('Construction Period (Years)', fontsize=12, fontweight='bold', color='#A23B72')
ax3_secondary.set_ylabel('Annual Carbon Emissions (10k tons CO2e)', fontsize=12, fontweight='bold', color='#F18F01')

ax1_primary.tick_params(axis='y', labelcolor='#2E86AB')
ax2_secondary.tick_params(axis='y', labelcolor='#A23B72')
ax3_secondary.tick_params(axis='y', labelcolor='#F18F01')

ax1_primary.set_xticks(x)
ax1_primary.set_xticklabels(scenarios, fontsize=12, fontweight='bold')

for bars, values, ax_type in [(bars1, costs, ax1_primary), 
                               (bars2, periods, ax2_secondary), 
                               (bars3, emissions, ax3_secondary)]:
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax_type.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:,.2f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

ax1_primary.grid(True, axis='y', alpha=0.3, linestyle='--')

lines1, labels1 = ax1_primary.get_legend_handles_labels()
lines2, labels2 = ax2_secondary.get_legend_handles_labels()
lines3, labels3 = ax3_secondary.get_legend_handles_labels()
ax1_primary.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
                   loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=11)

ax1_primary.set_title('Comprehensive Comparison of Lunar Colony Construction Transport Schemes', fontsize=16, fontweight='bold', pad=20)

ax1_primary.annotate('Mixed Scheme Advantages:\n• 14.9% cost saving vs rocket\n• 25.9% period reduction vs elevator\n• 73.7% emission reduction vs rocket',
                    xy=(2.1, periods[2]/2), xytext=(2.5, periods[2]*0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig('7_final_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("All individual charts generated successfully!")
print("Saved as:")
print("1. 1_3d_comparison.png - 3D Comparison Chart")
print("2. 2_radar_chart.png - Radar Chart")
print("3. 3_pareto_frontier.png - Pareto Frontier Chart")
print("4. 4_sensitivity_analysis.png - Sensitivity Analysis Chart")
print("5. 5_cost_benefit.png - Cost-Benefit Comparison Chart")
print("6. 6_stacked_area.png - Stacked Area Chart")
print("7. 7_final_comparison.png - Final Comparison Chart")