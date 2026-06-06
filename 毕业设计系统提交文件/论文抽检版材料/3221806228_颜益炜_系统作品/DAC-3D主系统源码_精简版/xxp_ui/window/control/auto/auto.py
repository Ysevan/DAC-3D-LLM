def auto_focus():
    direction = 1  # 初始移动方向
    step = 0.1
    current_value = clarity_evaluation()
    
    while step >= 0.001:
        # 直接移动不记录位置
        移动(step * direction)
        new_value = clarity_evaluation()
        
        if new_value > current_value:
            current_value = new_value  # 保持方向继续前进
        else:
            # 仅调整参数不执行回退
            step /= 2      # 步长减半
            direction *= -1  # 立即反转方向
            
    # 最终微调后保持当前位置
