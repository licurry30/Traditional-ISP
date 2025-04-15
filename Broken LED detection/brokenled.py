import cv2
import numpy as np

# 读取图像
gray_image = cv2.imread('src.jpg', cv2.IMREAD_GRAYSCALE)

# 去除噪声
blur_image = cv2.medianBlur(gray_image, 5)

# 图像尺寸
height, width = blur_image.shape

# 灯珠排列数量
num_rows = 58
num_cols = 59 
# 计算网格单元的大小
cell_height = (height // num_rows) * 2
cell_width = (width // num_cols) * 2
#print(cell_height,cell_width)

# 初始化一个列表存储检测到的灯珠信息
detected_circles = []
# 初始化一个列表存储每行第一个检测到的灯珠位置
first_lights_col= []  # 使用 -1 表示尚未检测到
first_lights_row = []  # 使用 -1 表示尚未检测到
r_use = 0 #半径
# 遍历每一行
for row in range(0, height, cell_height):
    # 遍历每一列
    for col in range(0, width, cell_width):
        # 计算网格单元的起始和结束坐标
        start_y = row
        end_y = min(start_y + cell_height, height)
        start_x = col
        end_x = min(start_x + cell_width, width)
        
        # 提取当前网格单元内的图像
        cell_image = blur_image[start_y:end_y, start_x:end_x]
        
        # 使用霍夫圆变换检测灯珠，dp=1，minDist=10，param1=50，param2=30，minRadius=5，maxRadius=15
        circles = cv2.HoughCircles(cell_image, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                   param1=50, param2=30, minRadius=5, maxRadius=50)
        
        # 如果检测到圆
        if circles is not None:
            # 将圆心坐标和半径转换为整数
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # 将圆心坐标转换为原图像中的坐标
                x += start_x
                y += start_y
                # 将检测到的灯珠信息添加到列表中
                detected_circles.append((x, y, r))
                r_use= r #保存待使用
                

# 将二值图像转换为BGR图像，以便在图像上绘制彩色圆
output_image = cv2.cvtColor(blur_image, cv2.COLOR_GRAY2BGR)

# 在图像上绘制检测到的圆，颜色为绿色，圆的粗细为2
for (x, y, r) in detected_circles:
    cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
    if y < 80:
        first_lights_row.append((x,y))
    if x < 80:
        first_lights_col.append((x,y))

dist_col = first_lights_col[1][1]-first_lights_col[0][1]
start_row = first_lights_row[0][1] - dist_col // 2       #调整起始行


# cv2.imwrite('detected_led.jpg',output_image)

# 初始化一个二维数组记录每个网格单元是否检测到灯珠
grid = np.zeros((num_rows, num_cols), dtype=int)
# 遍历所有检测到的灯珠
for (x, y, r) in detected_circles:
    # 计算灯珠所在的行和列
    row = y // (cell_height // 2)  # 使用原始网格大小计算
    col = x // (cell_width // 2)
    # 标记该网格单元内检测到灯珠
    grid[row, col] = 1

# 初始化一个列表存储缺失灯珠的位置
missing_positions = []
# 遍历每一行
for row in range(num_rows):
    # 遍历每一列
    for col in range(num_cols):
        # 如果该网格单元内没有检测到灯珠
        if grid[row, col] == 0:
            # 计算缺失灯珠的中心坐标
            missing_x = col * (cell_width // 2) + (cell_width // 2) // 2
            missing_y = start_row + row * (cell_height // 2) + (cell_height // 2) // 2
            # 将缺失灯珠的位置添加到列表中
            missing_positions.append((missing_x, missing_y)) #
            

for (x, y) in missing_positions:
    cv2.circle(output_image, (x, y), r_use , (0, 0, 255), 2)
    start_x = max(x - cell_width // 4, 0)
    end_x = min(x + cell_width // 4, width)
    start_y = max(y - cell_height // 4, 0)
    end_y = min(y + cell_height // 4, height)
    broken_led_image = gray_image[start_y:end_y, start_x:end_x]
    cv2.imwrite(f'broken_led_{x}_{y}.jpg', broken_led_image)

with open('coordinates.txt', 'w') as f:
    for (x, y) in missing_positions:
        f.write(f'({x}, {y})\n')

cv2.imwrite('premiss_led.jpg',output_image)
