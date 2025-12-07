from PIL import Image

# 打开 PNG 图片
image = Image.open("/inspire/hdd/global_user/chenxinyan-240108120066/yihang/LVSM/objaverse-xl/scripts/rendering/data_rendered/000-000/00a1a602456f4eb188b522d7ef19e81b/000.png")

# 获取图片的分辨率（宽度和高度）
width, height = image.size

print(f"图片分辨率: {width}x{height}")