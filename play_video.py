from IPython.display import HTML, display
from base64 import b64encode
import os

# 【关键修正】根据你的发现，我们直接从moviepy导入
from moviepy import ImageSequenceClip 

# --- 请在这里指定你的视频文件路径 ---
# (你需要根据左侧文件浏览器里的实际情况，填写准确的文件夹和文件名)
VIDEO_FOLDER = "marl_videos" 
VIDEO_FILENAME = "random-spread-episode-0.mp4" # <--- 示例文件名，请替换成你的
video_path = os.path.join(VIDEO_FOLDER, VIDEO_FILENAME)
# -----------------------------------------


print(f"正在加载视频文件: {video_path}")

if os.path.exists(video_path):
    with open(video_path, 'rb') as f:
        video_data = f.read()

    video_base64 = b64encode(video_data).decode('ascii')
    
    # 创建HTML播放器
    video_html = f"""
    <video width="600" height="400" controls autoplay loop>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
         Your browser does not support the video tag.
    </video>
    """
    
    # 在单元格输出区域显示视频
    display(HTML(video_html))
else:
    print(f"错误: 找不到视频文件 '{video_path}'")
    print("请确认上面的 FOLDER 和 FILENAME 变量是否填写正确。")