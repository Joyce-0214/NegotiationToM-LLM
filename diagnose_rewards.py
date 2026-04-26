#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本：分析NegotiationToM的reward计算问题
"""

import numpy as np

def analyze_margin_reward():
    """
    分析margin reward的计算逻辑
    
    Reward公式：
    - 失败: seller=-0.5, buyer=-0.5
    - 成功: reward = (price - midpoint) / norm_factor
      其中 midpoint = (seller_target + buyer_target) / 2
           norm_factor = |midpoint - seller_target|
    """
    
    print("="*60)
    print("Margin Reward 分析")
    print("="*60)
    
    # 模拟场景
    scenarios = [
        # (seller_target, buyer_target, final_price, description)
        (100, 70, 85, "理想成交 - 中点"),
        (100, 70, 100, "卖家完胜"),
        (100, 70, 70, "买家完胜"),
        (100, 70, 95, "偏向卖家"),
        (100, 70, 75, "偏向买家"),
        (100, 70, 110, "超出卖家目标"),
        (100, 70, 60, "低于买家目标"),
    ]
    
    print("\n场景分析:")
    print("-"*60)
    
    for seller_target, buyer_target, price, desc in scenarios:
        midpoint = (seller_target + buyer_target) / 2.0
        norm_factor = abs(midpoint - seller_target)
        
        seller_reward = (price - midpoint) / norm_factor
        buyer_reward = -1.0 * seller_reward
        
        print(f"\n{desc}:")
        print(f"  卖家目标: ${seller_target}, 买家目标: ${buyer_target}")
        print(f"  成交价格: ${price}")
        print(f"  中点: ${midpoint}, 归一化因子: {norm_factor}")
        print(f"  卖家reward: {seller_reward:.3f}")
        print(f"  买家reward: {buyer_reward:.3f}")
        
        if seller_reward < 0:
            print(f"  ⚠️  卖家获得负reward！价格低于中点")
        if buyer_reward < 0:
            print(f"  ⚠️  买家获得负reward！价格高于中点")

def analyze_your_results():
    """
    分析你的实际结果
    """
    print("\n" + "="*60)
    print("你的训练结果分析")
    print("="*60)
    
    success_rate = 0.5826
    avg_reward = -0.4020
    
    print(f"\n成功率: {success_rate*100:.1f}%")
    print(f"平均reward: {avg_reward:.4f}")
    
    # 反推成功案例的平均reward
    fail_rate = 1 - success_rate
    fail_reward = -0.5
    
    # avg_reward = fail_rate * fail_reward + success_rate * success_reward
    # success_reward = (avg_reward - fail_rate * fail_reward) / success_rate
    
    success_reward = (avg_reward - fail_rate * fail_reward) / success_rate
    
    print(f"\n推算:")
    print(f"  失败率: {fail_rate*100:.1f}% × reward(-0.5) = {fail_rate * fail_reward:.4f}")
    print(f"  成功率: {success_rate*100:.1f}% × reward(?) = ?")
    print(f"  推算成功案例平均reward: {success_reward:.4f}")
    
    print(f"\n⚠️  关键发现:")
    print(f"  即使成交了，平均reward仍然是负数({success_reward:.4f})！")
    print(f"  这说明成交价格严重偏离中点。")
    
    # 反推价格偏离
    # success_reward = (price - midpoint) / norm_factor
    # 假设 norm_factor = (seller_target - buyer_target) / 2
    # 如果 success_reward = -0.33，说明 price 比 midpoint 低 0.33 * norm_factor
    
    print(f"\n  如果卖家的成功reward是{success_reward:.4f}:")
    print(f"    意味着成交价格 = 中点 + {success_reward:.2f} × 归一化因子")
    print(f"    即成交价格比中点低了约{-success_reward*100:.0f}%的价格范围")
    print(f"    这对卖家非常不利！")

def analyze_price_action_bug():
    """
    分析价格动作映射的bug
    """
    print("\n" + "="*60)
    print("价格动作映射Bug分析")
    print("="*60)
    
    print("\n当前实现 (_pact_to_price):")
    print("  p_act=0: insist (pmax)")
    print("  p_act=1: agree (pmin)")
    print("  p_act=2: middle ((pmax+pmin)/2)")
    print("  p_act=3: decay (pmax - 0.1)  ← BUG!")
    
    print("\n问题场景:")
    scenarios = [
        (0.8, 0.6, "正常范围"),
        (0.9, 0.85, "窄范围"),
        (0.3, 0.1, "低价范围"),
    ]
    
    for pmax, pmin, desc in scenarios:
        p_decay = pmax - 0.1
        p_decay_clamped = max(min(p_decay, pmax), pmin)
        
        print(f"\n  {desc}: pmax={pmax}, pmin={pmin}")
        print(f"    decay动作: {pmax} - 0.1 = {p_decay}")
        
        if p_decay < pmin:
            print(f"    ⚠️  超出下界！实际应该是{pmin}")
        elif p_decay > pmax:
            print(f"    ⚠️  超出上界！")
        
        # 正确的相对衰减
        p_decay_correct = pmax - 0.1 * (pmax - pmin)
        print(f"    正确的相对衰减: {pmax} - 0.1×({pmax}-{pmin}) = {p_decay_correct:.3f}")

def check_price_scaling():
    """
    检查价格缩放问题
    """
    print("\n" + "="*60)
    print("价格缩放检查")
    print("="*60)
    
    print("\n价格缩放公式:")
    print("  scaled = w * real + c")
    print("  其中 w = 1/(target-bottom), c = -bottom/(target-bottom)")
    print("\n反缩放公式:")
    print("  real = (scaled - c) / w")
    print("  然后 round(real)  ← 可能有问题！")
    
    print("\n测试场景:")
    scenarios = [
        (70, 100, 85.4, "正常情况"),
        (70, 100, 85.6, "接近整数"),
        (95, 100, 97.3, "窄范围"),
    ]
    
    for bottom, target, real_price, desc in scenarios:
        w = 1.0 / (target - bottom)
        c = -bottom / (target - bottom)
        
        scaled = w * real_price + c
        unscaled = (scaled - c) / w
        rounded = round(unscaled)
        
        error = abs(real_price - rounded)
        error_pct = error / (target - bottom) * 100
        
        print(f"\n  底价={bottom}, 目标={target}, 真实价格={real_price}")
        print(f"    缩放后: {scaled:.4f}")
        print(f"    反缩放: {unscaled:.4f}")
        print(f"    四舍五入: {rounded}")
        print(f"    误差: {error:.2f} ({error_pct:.1f}%)")
        
        if error_pct > 5:
            print(f"    ⚠️  误差超过5%！可能导致谈判失败")

if __name__ == "__main__":
    analyze_margin_reward()
    analyze_your_results()
    analyze_price_action_bug()
    check_price_scaling()
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("""
主要问题:
1. ⚠️  成功案例的平均reward仍然是负数(-0.33)
   → 说明成交价格严重偏离中点，对某一方非常不利

2. ⚠️  价格动作映射使用绝对值(-0.1)而非相对值
   → 在归一化空间[0,1]中，0.1可能是巨大的变化
   → 没有边界检查，可能超出有效范围

3. ⚠️  价格四舍五入可能引入误差
   → 在窄价格范围时，误差百分比很大

4. ⚠️  58.3%的成功率偏低
   → 论文中通常是70-90%
   → 说明agent策略有问题

建议修复顺序:
1. 修复 _pact_to_price() 使用相对衰减
2. 添加边界检查
3. 检查价格缩放的舍入误差
4. 添加详细日志追踪实际价格
""")
