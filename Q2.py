# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 23:34:54 2026

@author: admin
"""

"""
Lunar Colony Logistics Optimization Model Visualization - English Version
Display all charts in English directly on screen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Set English fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Color scheme
COLOR_SCHEME = [
    '#6AD1A3',  # Green
    '#7FBDDA',  # Blue
    '#BBC7BE',  # Gray-green
    '#FFD47D',  # Yellow
    '#FFA288',  # Orange
    '#C49892',  # Brown
    '#929EAB',  # Gray
    '#84ADDC'   # Light blue
]

def show_visualizations():
    """Display all visualization charts on screen"""
    
    # 1. Cost Comparison Chart
    print("Displaying Chart 1: Cost Comparison...")
    plt.figure(figsize=(10, 6))
    
    scenarios = ['Space Elevator', 'Rocket Only', 'Hybrid']
    base_costs = [50.00, 62.99, 53.62]  # Table data
    risk_costs = [53.75, 70.55, 56.84]  # Risk-adjusted cost
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, base_costs, width, label='Base Cost', 
                   color=COLOR_SCHEME[1], edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x + width/2, risk_costs, width, label='Risk-adjusted Cost', 
                   color=COLOR_SCHEME[4], edgecolor='black', linewidth=1.5)
    
    plt.xlabel('Transportation Scenario', fontsize=12, fontweight='bold')
    plt.ylabel('Cost (Trillion USD)', fontsize=12, fontweight='bold')
    plt.title('Cost Comparison: Base vs Risk-adjusted', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, scenarios, fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, 
                    f'${height:.2f}T', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Construction Time Comparison Chart
    print("Displaying Chart 2: Construction Time Comparison...")
    plt.figure(figsize=(10, 6))
    
    construction_times = [647.8, 1978.2, 474.1]
    
    bars = plt.bar(scenarios, construction_times, 
                  color=[COLOR_SCHEME[0], COLOR_SCHEME[4], COLOR_SCHEME[1]],
                  edgecolor='black', linewidth=1.5)
    
    plt.xlabel('Transportation Scenario', fontsize=12, fontweight='bold')
    plt.ylabel('Construction Period (Years)', fontsize=12, fontweight='bold')
    plt.title('Construction Time Comparison (Non-ideal Conditions)', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bar, time_val in zip(bars, construction_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, 
                f'{time_val:.1f} years', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        
        # Add "shortest" label
        if time_val == min(construction_times):
            plt.text(bar.get_x() + bar.get_width()/2, height*1.02, 
                    'Shortest', ha='center', va='bottom', 
                    fontsize=9, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 3. Carbon Emissions Comparison Chart
    print("Displaying Chart 3: Carbon Emissions Comparison...")
    plt.figure(figsize=(10, 6))
    
    emissions = [0.00, 7874.02, 2070.39]
    
    bars = plt.bar(scenarios, emissions, 
                  color=[COLOR_SCHEME[2], COLOR_SCHEME[5], COLOR_SCHEME[0]],
                  edgecolor='black', linewidth=1.5)
    
    plt.xlabel('Transportation Scenario', fontsize=12, fontweight='bold')
    plt.ylabel('Annual Carbon Emissions (10k tons CO₂e)', fontsize=12, fontweight='bold')
    plt.title('Carbon Emissions Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bar, emission_val in zip(bars, emissions):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2, height, 
                    f'{emission_val:.1f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*0.5, 
                    '0.00', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Reliability Comparison Chart
    print("Displaying Chart 4: System Reliability Comparison...")
    plt.figure(figsize=(10, 6))
    
    reliabilities = [92.0, 87.2, 92.8]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(scenarios))
    
    bars = plt.barh(y_pos, reliabilities, 
                   color=[COLOR_SCHEME[0], COLOR_SCHEME[4], COLOR_SCHEME[1]],
                   edgecolor='black', linewidth=1.5)
    
    plt.xlabel('System Reliability (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Transportation Scenario', fontsize=12, fontweight='bold')
    plt.title('System Reliability Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.yticks(y_pos, scenarios, fontsize=11)
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels
    for bar, reliability_val in zip(bars, reliabilities):
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{reliability_val:.1f}%', ha='left', va='center', 
                fontsize=10, fontweight='bold')
        
        # Add label inside bar
        if width > 20:
            plt.text(width/2, bar.get_y() + bar.get_height()/2, 
                    f'{reliability_val:.1f}%', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Hybrid Scheme Transportation Ratio Chart
    print("Displaying Chart 5: Hybrid Scheme Transportation Ratio...")
    plt.figure(figsize=(8, 8))
    
    labels = ['Space Elevator (72.1%)', 'Rocket (27.9%)']
    sizes = [72.1, 27.9]
    colors = [COLOR_SCHEME[0], COLOR_SCHEME[4]]
    explode = (0.05, 0)  # Highlight first part
    
    wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90, 
                                      textprops={'fontsize': 12, 'fontweight': 'bold'},
                                      wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    
    # Style percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    plt.title('Hybrid Transportation Scheme Ratio', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    plt.legend(wedges, labels, title="Transportation Method", loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Sensitivity Analysis Chart
    print("Displaying Chart 6: Sensitivity Analysis...")
    plt.figure(figsize=(10, 6))
    
    # Simulate failure rate impact on construction time
    pf_values = [0.005, 0.01, 0.02, 0.03, 0.05]
    # Assume construction time increases linearly with failure rate
    base_time = 647.8
    time_impact = [base_time * (1 + 0.1*p/pf_values[1]) for p in pf_values]
    
    plt.plot(pf_values, time_impact, marker='o', linewidth=3, markersize=8,
            color=COLOR_SCHEME[1], label='Construction Time')
    
    plt.xlabel('Space Elevator Failure Rate (pf)', fontsize=12, fontweight='bold')
    plt.ylabel('Construction Period (Years)', fontsize=12, fontweight='bold')
    plt.title('Impact of Failure Rate on Construction Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Mark baseline point
    baseline_idx = pf_values.index(0.01)
    plt.scatter(0.01, time_impact[baseline_idx], s=150, 
               color=COLOR_SCHEME[4], zorder=5, edgecolors='black', linewidth=2)
    plt.annotate('Baseline\n(pf=0.01)', xy=(0.01, time_impact[baseline_idx]), 
                xytext=(0.02, time_impact[baseline_idx] + 50),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor=COLOR_SCHEME[7], alpha=0.8))
    
    # Add sensitivity index label
    sensitivity_index = 0.001
    plt.text(0.03, time_impact[-1]*0.9, 
            f'Sensitivity Index: {sensitivity_index:.3f}\n(Low Sensitivity)', 
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_SCHEME[6], alpha=0.8))
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # 7. Comprehensive Evaluation Radar Chart
    print("Displaying Chart 7: Comprehensive Evaluation Radar Chart...")
    plt.figure(figsize=(8, 8))
    
    # Radar chart data
    categories = ['Cost Efficiency', 'Time Efficiency', 'Environmental Friendliness', 'System Reliability', 'Risk Control']
    
    # Normalized data (0-1 range)
    space_elevator = [0.8, 0.7, 1.0, 0.92, 0.85]  # Space Elevator Only
    rocket_only = [0.6, 0.3, 0.2, 0.87, 0.75]     # Rocket Only
    hybrid = [0.9, 0.9, 0.8, 0.93, 0.88]          # Hybrid
    
    # Close radar chart
    space_elevator += space_elevator[:1]
    rocket_only += rocket_only[:1]
    hybrid += hybrid[:1]
    categories += categories[:1]
    
    # Create angles
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
    
    # Create polar chart
    ax = plt.subplot(111, polar=True)
    
    # Draw three scenarios
    ax.plot(angles, space_elevator, 'o-', linewidth=2, markersize=6,
           color=COLOR_SCHEME[0], label='Space Elevator')
    ax.fill(angles, space_elevator, alpha=0.25, color=COLOR_SCHEME[0])
    
    ax.plot(angles, rocket_only, 'o-', linewidth=2, markersize=6,
           color=COLOR_SCHEME[4], label='Rocket Only')
    ax.fill(angles, rocket_only, alpha=0.25, color=COLOR_SCHEME[4])
    
    ax.plot(angles, hybrid, 'o-', linewidth=2, markersize=6,
           color=COLOR_SCHEME[1], label='Hybrid')
    ax.fill(angles, hybrid, alpha=0.25, color=COLOR_SCHEME[1])
    
    # Set angle labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=11, fontweight='bold')
    
    # Set radial labels
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    
    # Add title
    plt.title('Comprehensive Evaluation Radar Chart', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    # 8. Comprehensive Comparison Bar Chart
    print("Displaying Chart 8: Comprehensive Comparison Bar Chart...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Prepare data
    metrics = ['Cost\n(Trillion USD)', 'Time\n(Years)', 'Emissions\n(10k tons)', 'Reliability\n(%)']
    space_data = [50.00, 647.8, 0.0, 92.0]
    rocket_data = [62.99, 1978.2, 7874.02, 87.2]
    hybrid_data = [53.62, 474.1, 2070.39, 92.8]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    # Subplot 1: Cost Comparison
    axs[0, 0].bar(x[0] - width, space_data[0], width, label='Space Elevator', color=COLOR_SCHEME[0])
    axs[0, 0].bar(x[0], rocket_data[0], width, label='Rocket Only', color=COLOR_SCHEME[4])
    axs[0, 0].bar(x[0] + width, hybrid_data[0], width, label='Hybrid', color=COLOR_SCHEME[1])
    axs[0, 0].set_title('Cost Comparison', fontsize=12, fontweight='bold')
    axs[0, 0].set_ylabel('Trillion USD', fontsize=10)
    axs[0, 0].set_xticks([x[0]])
    axs[0, 0].set_xticklabels([metrics[0]])
    axs[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Time Comparison
    axs[0, 1].bar(x[1] - width, space_data[1], width, color=COLOR_SCHEME[0])
    axs[0, 1].bar(x[1], rocket_data[1], width, color=COLOR_SCHEME[4])
    axs[0, 1].bar(x[1] + width, hybrid_data[1], width, color=COLOR_SCHEME[1])
    axs[0, 1].set_title('Construction Time', fontsize=12, fontweight='bold')
    axs[0, 1].set_ylabel('Years', fontsize=10)
    axs[0, 1].set_xticks([x[1]])
    axs[0, 1].set_xticklabels([metrics[1]])
    axs[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Emissions Comparison
    axs[1, 0].bar(x[2] - width, space_data[2], width, color=COLOR_SCHEME[0])
    axs[1, 0].bar(x[2], rocket_data[2], width, color=COLOR_SCHEME[4])
    axs[1, 0].bar(x[2] + width, hybrid_data[2], width, color=COLOR_SCHEME[1])
    axs[1, 0].set_title('Carbon Emissions', fontsize=12, fontweight='bold')
    axs[1, 0].set_ylabel('10k tons CO₂e', fontsize=10)
    axs[1, 0].set_xticks([x[2]])
    axs[1, 0].set_xticklabels([metrics[2]])
    axs[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Reliability Comparison
    axs[1, 1].bar(x[3] - width, space_data[3], width, color=COLOR_SCHEME[0])
    axs[1, 1].bar(x[3], rocket_data[3], width, color=COLOR_SCHEME[4])
    axs[1, 1].bar(x[3] + width, hybrid_data[3], width, color=COLOR_SCHEME[1])
    axs[1, 1].set_title('System Reliability', fontsize=12, fontweight='bold')
    axs[1, 1].set_ylabel('%', fontsize=10)
    axs[1, 1].set_xticks([x[3]])
    axs[1, 1].set_xticklabels([metrics[3]])
    axs[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add overall legend
    fig.legend(['Space Elevator', 'Rocket Only', 'Hybrid'], 
               loc='upper center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.05))
    
    plt.suptitle('Comprehensive Scenario Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # 9. Time Series Prediction Chart
    print("Displaying Chart 9: Time Series Prediction...")
    plt.figure(figsize=(12, 7))
    
    # Create different time points for each scenario
    years_space = np.linspace(2026, 2673, 6)  # 6 points
    years_rocket = np.linspace(2026, 4004, 6)  # 6 points
    years_hybrid = np.linspace(2026, 2500, 6)  # 6 points
    
    # Simulate cumulative transportation volume (6 points)
    space_cumulative = [0, 0.2e8, 0.4e8, 0.6e8, 0.8e8, 1.0e8]
    rocket_cumulative = [0, 0.1e8, 0.3e8, 0.5e8, 0.8e8, 1.0e8]
    hybrid_cumulative = [0, 0.3e8, 0.6e8, 0.8e8, 0.95e8, 1.0e8]
    
    plt.plot(years_space, space_cumulative, 's-', linewidth=3, markersize=8,
            color=COLOR_SCHEME[0], label='Space Elevator', alpha=0.9)
    plt.plot(years_rocket, rocket_cumulative, '^-', linewidth=3, markersize=8,
            color=COLOR_SCHEME[4], label='Rocket Only', alpha=0.9)
    plt.plot(years_hybrid, hybrid_cumulative, 'o-', linewidth=3, markersize=8,
            color=COLOR_SCHEME[1], label='Hybrid', alpha=0.9)
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Transportation Volume (10k tons)', fontsize=12, fontweight='bold')
    plt.title('Time Series Transportation Progress Prediction', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='lower right')
    
    # Mark completion points
    completion_points = [(2673, 1.0e8), (4004, 1.0e8), (2500, 1.0e8)]
    labels = [f'Completed: {int(2673)}', f'Completed: {int(4004)}', f'Completed: {int(2500)}']
    colors = [COLOR_SCHEME[0], COLOR_SCHEME[4], COLOR_SCHEME[1]]
    
    for i, (x, y) in enumerate(completion_points):
        plt.scatter(x, y, s=150, color=colors[i], zorder=5, 
                   edgecolors='black', linewidth=2)
        plt.annotate(labels[i], xy=(x, y), xytext=(x-100, y*0.9),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 10. Comprehensive Score Chart
    print("Displaying Chart 10: Comprehensive Score Comparison...")
    plt.figure(figsize=(10, 6))
    
    categories = ['Cost', 'Time', 'Emissions', 'Reliability', 'Risk', 'Total Score']
    space_scores = [85, 70, 100, 92, 85, 432]  # Space Elevator
    rocket_scores = [60, 30, 20, 87, 75, 272]  # Rocket Only
    hybrid_scores = [90, 90, 80, 93, 88, 441]  # Hybrid
    
    x = np.arange(len(categories))
    width = 0.25
    
    plt.bar(x - width, space_scores, width, label='Space Elevator', 
           color=COLOR_SCHEME[0], edgecolor='black', linewidth=1.5)
    plt.bar(x, rocket_scores, width, label='Rocket Only', 
           color=COLOR_SCHEME[4], edgecolor='black', linewidth=1.5)
    plt.bar(x + width, hybrid_scores, width, label='Hybrid', 
           color=COLOR_SCHEME[1], edgecolor='black', linewidth=1.5)
    
    plt.xlabel('Evaluation Metric', fontsize=12, fontweight='bold')
    plt.ylabel('Score (0-100)', fontsize=12, fontweight='bold')
    plt.title('Comprehensive Score Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, categories, fontsize=11)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Mark highest score
    for i, cat in enumerate(categories):
        scores = [space_scores[i], rocket_scores[i], hybrid_scores[i]]
        max_score = max(scores)
        if max_score == hybrid_scores[i]:
            plt.text(i + width, hybrid_scores[i] + 2, 'Best', 
                    ha='center', fontsize=9, fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.show()
    
    # 11. Risk Cost Comparison Chart
    print("Displaying Chart 11: Risk Cost Comparison...")
    plt.figure(figsize=(10, 6))
    
    scenarios = ['Space Elevator', 'Rocket Only', 'Hybrid']
    base_costs = [50.00, 62.99, 53.62]  # Base cost
    risk_costs = [53.75, 70.55, 56.84]  # Risk-adjusted cost
    risk_increase = [risk_costs[i] - base_costs[i] for i in range(3)]
    
    x = np.arange(len(scenarios))
    width = 0.6
    
    # Create stacked bar chart
    p1 = plt.bar(x, base_costs, width, label='Base Cost', 
                color=COLOR_SCHEME[1], edgecolor='black', linewidth=1.5)
    p2 = plt.bar(x, risk_increase, width, bottom=base_costs, 
                label='Risk Cost Increment', color=COLOR_SCHEME[4], edgecolor='black', linewidth=1.5)
    
    plt.xlabel('Transportation Scenario', fontsize=12, fontweight='bold')
    plt.ylabel('Cost (Trillion USD)', fontsize=12, fontweight='bold')
    plt.title('Cost Structure: Base Cost + Risk Cost Increment', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, scenarios, fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add total cost labels
    for i, (base, risk) in enumerate(zip(base_costs, risk_costs)):
        plt.text(i, risk + 1, f'${risk:.2f}T', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        # Add percentage labels
        risk_pct = (risk - base) / base * 100
        plt.text(i, base + (risk - base)/2, f'+{risk_pct:.1f}%', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.show()
    
    # 12. Scenario Selection Decision Diagram
    print("Displaying Chart 12: Scenario Selection Decision Diagram...")
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot: x-axis = cost, y-axis = time
    costs = [53.75, 70.55, 56.84]  # Risk-adjusted cost
    times = [647.8, 1978.2, 474.1]  # Construction time
    emissions = [0.00, 7874.02, 2070.39]  # Emissions
    reliabilities = [92.0, 87.2, 92.8]  # Reliability
    
    # Point size represents reliability
    sizes = [r*20 for r in reliabilities]
    
    # Scatter plot
    scatter = plt.scatter(costs, times, s=sizes, c=emissions, 
                         cmap='viridis', alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Add scenario labels
    for i, (scenario, cost, time) in enumerate(zip(scenarios, costs, times)):
        plt.annotate(scenario, xy=(cost, time), xytext=(cost+1, time+50),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))
    
    # Add ideal region
    ideal_cost = min(costs)
    ideal_time = min(times)
    plt.axvline(x=ideal_cost, color='green', linestyle='--', alpha=0.5, label='Ideal Cost')
    plt.axhline(y=ideal_time, color='blue', linestyle='--', alpha=0.5, label='Ideal Time')
    
    # Mark optimal scenario
    plt.scatter(costs[2], times[2], s=200, color=COLOR_SCHEME[1], 
               edgecolors='red', linewidth=3, zorder=5, label='Recommended')
    
    plt.xlabel('Risk-adjusted Cost (Trillion USD)', fontsize=12, fontweight='bold')
    plt.ylabel('Construction Time (Years)', fontsize=12, fontweight='bold')
    plt.title('Scenario Selection: Cost vs Time vs Emissions vs Reliability', fontsize=14, fontweight='bold', pad=20)
    
    # Add color bar for emissions
    cbar = plt.colorbar(scatter)
    cbar.set_label('Carbon Emissions (10k tons CO₂e)', fontsize=11, fontweight='bold')
    
    # Add legend
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nAll visualization charts displayed!")
    print(f"Total of 12 charts displayed:")
    print("1. Cost Comparison Chart")
    print("2. Construction Time Comparison Chart")
    print("3. Carbon Emissions Comparison Chart")
    print("4. System Reliability Comparison Chart")
    print("5. Hybrid Scheme Transportation Ratio Chart")
    print("6. Sensitivity Analysis Chart")
    print("7. Comprehensive Evaluation Radar Chart")
    print("8. Comprehensive Comparison Bar Chart")
    print("9. Time Series Prediction Chart")
    print("10. Comprehensive Score Chart")
    print("11. Risk Cost Comparison Chart")
    print("12. Scenario Selection Decision Diagram")

# Main program
if __name__ == "__main__":
    print("="*60)
    print("Lunar Colony Logistics Optimization Model Visualization")
    print("="*60)
    print("Displaying 12 visualization charts on screen...")
    print("Close each window to see the next chart")
    print("="*60)
    
    try:
        # Display all visualization charts
        show_visualizations()
        
        print("\n" + "="*60)
        print("All charts displayed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during display: {e}")
        import traceback
        traceback.print_exc()