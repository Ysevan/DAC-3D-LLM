import base64

mermaid_code = """
graph TD
    subgraph "表现层 (Presentation Layer)"
        UI[主控制界面 MyWindow]
        Login[登录验证 Login]
        Debug[硬件调试界面 Debug_UI]
    end

    subgraph "业务逻辑与进程管理"
        Main[主控程序 main.py]
        Queue1[UI-Debug 消息队列]
        Queue2[UI-Image 图像队列]
        Main -->|启动进程| UI
        Main -->|启动进程| Debug
        Main -->|启动进程| IP[后端图像处理进程]
        UI <--> Queue1 <--> Debug
        UI <--> Queue2 <--> IP
    end

    subgraph "算法层 (Algorithm Layer)"
        subgraph "传统视觉算法"
            Reg[图像配准 Registration]
            Fusion[共聚焦融合 Fusion]
            Fit[RANSAC 圆形拟合]
        end
        subgraph "深度学习模型"
            YOLO[YOLO 缺陷识别]
            SAHI[SAHI 切片推理]
        end
        IP --> Reg
        IP --> Fusion
        IP --> Fit
        IP --> YOLO
        IP --> SAHI
    end

    subgraph "硬件驱动与数据支撑"
        Camera[海康相机 SDK]
        Motion[正运动控制器 DLL]
        Light[光源控制协议]
        MySQL[(MySQL 数据库)]
        UI -->|控制指令| Camera
        UI -->|运动指令| Motion
        UI -->|亮度调节| Light
        UI -->|历史读写| MySQL
    end
"""

# Base64 encoding for Mermaid.ink
encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
image_url = f"https://mermaid.ink/img/{encoded}"
print(f"Image URL: {image_url}")

# Also create an HTML file for local viewing
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Software Architecture Diagram</title>
</head>
<body>
    <h1>软件架构图 (Software Architecture Diagram)</h1>
    <p>你可以直接右键保存下方的图片：</p>
    <img src="{image_url}" alt="Software Architecture">
    <br>
    <p>如果图片未显示，请访问: <a href="{image_url}" target="_blank">{image_url}</a></p>
</body>
</html>
"""

with open("arch_diagram.html", "w", encoding="utf-8") as f:
    f.write(html_content)
    print("HTML file 'arch_diagram.html' has been created.")
