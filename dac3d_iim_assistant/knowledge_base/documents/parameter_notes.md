# DAC-3D 参数说明 / Parameter Notes

## 分辨率 / Resolution
Resolution 控制扫描时的点间距。点间距越小，细节越丰富，但扫描时间和数据量也会增加。若只是做预览或初步确认，不一定需要高精度分辨率。

## 扫描区域 / Scan Area
Scan area 定义被检测区域的物理长宽，单位通常为毫米。扫描区域越大，循环时间越长，只有在缺陷位置不明确时才应扩大范围。

## 区域 / Region
Region 表示当前要采集的样品区域。安全做法是绑定到可见选区或已保存的感兴趣区域，不要在没有确认选区的情况下执行整片扫描。

## 模式 / Mode
Fast mode 适合预览，standard mode 适合常规检测，precision mode 用于需要更高细节的缺陷确认。切换到 precision mode 前，应先确认 standard mode 的结果是否已经足够。
