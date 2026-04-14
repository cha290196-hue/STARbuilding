import cv2
import numpy as np
from pathlib import Path
import json
from segment_anything import sam_model_registry, SamPredictor
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QMessageBox, 
                             QGroupBox, QSlider, QSpinBox, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsItem,
                             QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsRectItem,
                             QFrame, QGraphicsSimpleTextItem, QRadioButton, QButtonGroup,
                             QGraphicsLineItem, QComboBox, QSplitter, QSizePolicy,
                             QListWidget, QListWidgetItem, QLineEdit, QTabWidget, QScrollArea,
                             QGridLayout, QTreeWidget, QTreeWidgetItem, QAbstractItemView,
                             QHeaderView, QTextEdit, QDialog, QProgressBar, QInputDialog,
                             QMenu, QAction, QToolBar, QDockWidget, QToolButton, QCheckBox)
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent, 
                         QPainterPath, QBrush, QPolygonF, QTransform, QFont,
                         QPainterPath, QKeyEvent, QCursor, QPainterPathStroker, QIcon)
from PyQt5.QtCore import Qt, QPoint, QRect, QPointF, QSize, pyqtSignal, QLineF, QRectF, pyqtSlot, QObject, QThread, pyqtSignal as Signal
import sys
import os
import warnings
import copy
import threading
import time
from collections import OrderedDict
from functools import lru_cache
warnings.filterwarnings('ignore')

# 获取资源文件路径
def resource_path(relative_path):
    """获取资源文件的绝对路径"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller打包后的环境
        return os.path.join(sys._MEIPASS, relative_path)
    # 开发环境
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

# 图像缓存管理器
class ImageCache:
    """图像缓存管理器"""
    
    _instance = None
    _cache = OrderedDict()
    _max_size = 100  # 增加最大缓存图像数
    _lock = threading.Lock()
    _memory_usage = 0  # 当前缓存内存使用量（MB）
    _max_memory_mb = 500  # 最大内存使用量（MB）
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get(cls, image_path):
        """获取缓存的图像"""
        with cls._lock:
            if image_path in cls._cache:
                # 移动到最近使用
                cls._cache.move_to_end(image_path)
                return cls._cache[image_path].copy()
        return None
    
    @classmethod
    def set(cls, image_path, image_data):
        """缓存图像"""
        with cls._lock:
            # 计算图像大小（MB）
            image_size_mb = image_data.nbytes / (1024 * 1024)
            
            if image_path in cls._cache:
                # 更新缓存项
                old_size = cls._cache[image_path].nbytes / (1024 * 1024)
                cls._memory_usage -= old_size
                cls._cache[image_path] = image_data.copy()
                cls._memory_usage += image_size_mb
                cls._cache.move_to_end(image_path)
            else:
                # 确保有足够的内存
                while len(cls._cache) >= cls._max_size or cls._memory_usage + image_size_mb > cls._max_memory_mb:
                    # 移除最久未使用的图像
                    old_path, old_data = cls._cache.popitem(last=False)
                    cls._memory_usage -= old_data.nbytes / (1024 * 1024)
                
                # 添加新图像到缓存
                cls._cache[image_path] = image_data.copy()
                cls._memory_usage += image_size_mb
    
    @classmethod
    def clear(cls):
        """清空缓存"""
        with cls._lock:
            cls._cache.clear()
            cls._memory_usage = 0
    
    @classmethod
    def preload(cls, image_paths, load_func):
        """预加载图像"""
        for image_path in image_paths:
            if image_path not in cls._cache:
                try:
                    image_data = load_func(image_path)
                    if image_data is not None:
                        cls.set(image_path, image_data)
                except Exception as e:
                    print(f"预加载图像失败 {image_path}: {e}")
    
    @classmethod
    def remove(cls, image_path):
        """移除特定图像缓存"""
        with cls._lock:
            if image_path in cls._cache:
                old_size = cls._cache[image_path].nbytes / (1024 * 1024)
                cls._memory_usage -= old_size
                del cls._cache[image_path]

# 图像加载线程
class ImageLoadThread(QThread):
    """图像加载线程"""
    image_loaded = Signal(str, np.ndarray)  # 图像路径, 图像数据
    loading_progress = Signal(str, int)  # 状态信息, 进度
    
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
    
    def run(self):
        """运行线程"""
        try:
            # 检查缓存
            cached_image = ImageCache.get(self.image_path)
            if cached_image is not None:
                self.loading_progress.emit(f"从缓存加载: {Path(self.image_path).name}", 100)
                self.image_loaded.emit(self.image_path, cached_image)
                return
            
            # 开始加载图像
            self.loading_progress.emit(f"加载图像: {Path(self.image_path).name}", 20)

            # 支持中文路径的图像加载方法
            def imread_chinese(path, mode=cv2.IMREAD_COLOR):
                """读取带中文路径的图像"""
                import numpy as np
                with open(path, 'rb') as f:
                    image_data = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(image_data, mode)
                return img

            # 加载原始图像，不进行任何尺寸或颜色空间的改变
            img = imread_chinese(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                # 如果加载失败，尝试使用保持原始格式的方式加载
                img = imread_chinese(self.image_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"无法读取图像: {self.image_path}")
            
            # 转换为RGB
            self.loading_progress.emit(f"处理图像: {Path(self.image_path).name}", 60)
            image_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 缓存图像
            ImageCache.set(self.image_path, image_array)

            # 完成加载
            self.loading_progress.emit(f"加载完成: {Path(self.image_path).name}", 100)
            self.image_loaded.emit(self.image_path, image_array)
            
        except Exception as e:
            print(f"图像加载失败: {e}")
            # 发送错误信号
            self.image_loaded.emit(self.image_path, None)

class PixelPrecisionSAM:
    """像素级精度的SAM标注器"""
    
    def __init__(self, model_type="vit_b", checkpoint_path=None):
        print("正在初始化SAM模型...")
        
        # 设置设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 图像数据
        self.image = None
        self.image_path = None
        
        # 标注数据
        self.buildings = []  # 存储所有建筑物实例
        self.current_building_id = 0
        
        # SAM参数
        self.prediction_confidence = 0.5
        self.precision_level = 5  # 新增：多边形精细度，范围1-10，默认5
        
        # 预测器 - 如果没有提供，将在后续设置
        self.predictor = None
    
    def set_precision_level(self, level):
        """设置多边形精细度 (1-10)"""
        self.precision_level = max(1, min(10, level))
        print(f"设置多边形精细度为: {self.precision_level}")
    
    def set_image(self, image_path):
        """设置当前图像"""
        # 检查缓存
        cached_image = ImageCache.get(image_path)
        if cached_image is not None:
            print(f"从缓存加载图像: {image_path}")
            img = cached_image
        else:
            # 支持中文路径的图像加载方法
            def imread_chinese(path, mode=cv2.IMREAD_COLOR):
                """读取带中文路径的图像"""
                import numpy as np
                with open(path, 'rb') as f:
                    image_data = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(image_data, mode)
                return img
            
            img = imread_chinese(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 缓存图像
            ImageCache.set(image_path, img)
            print(f"从磁盘加载图像: {image_path}")
        
        self.image = img
        self.image_path = image_path
        
        # 重置预测器并设置图像
        self.predictor.set_image(self.image)
        
        # 重置标注数据
        self.buildings = []
        self.current_building_id = 0
        
        print(f"图像已加载: {image_path}, 尺寸: {self.image.shape}")
        return self.image
    
    def segment_building(self, x, y):
        """在指定位置分割建筑物（像素级精度）"""
        if self.image is None:
            print("错误: 未加载图像")
            return None
        
        try:
            # 使用SAM预测（使用多点输入获得更好效果）
            point_coords = np.array([[x, y]])
            point_labels = np.array([1])  # 正点
            
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
                mask_input=None
            )
            
            # 选择最佳掩码
            best_idx = self._select_best_mask(masks, scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            
            if best_score < self.prediction_confidence:
                print(f"置信度过低: {best_score:.3f} < {self.prediction_confidence}")
                return None
            
            # 获取精确的轮廓（像素级）
            contours = self._get_pixel_precision_contour(best_mask)
            
            if not contours:
                print("未找到轮廓")
                return None
            
            # 获取主要轮廓
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            
            if area < 50:  # 最小面积
                print(f"面积过小: {area:.0f}")
                return None
            
            # 根据精细度调整轮廓近似参数
            # 精细度越高，epsilon越小，保留更多点；精细度越低，epsilon越大，简化更多
            if self.precision_level >= 8:  # 高精细度
                epsilon = 0.001 * cv2.arcLength(main_contour, True)  # 保留更多细节
                max_points = 200  # 允许更多点
            elif self.precision_level >= 5:  # 中等精细度
                epsilon = 0.002 * cv2.arcLength(main_contour, True)
                max_points = 100
            elif self.precision_level >= 3:  # 低精细度
                epsilon = 0.005 * cv2.arcLength(main_contour, True)
                max_points = 50
            else:  # 最低精细度
                epsilon = 0.01 * cv2.arcLength(main_contour, True)
                max_points = 30
            
            # 简化轮廓
            approx = cv2.approxPolyDP(main_contour, epsilon, True)
            
            # 限制点数，太多点会导致界面卡顿
            if len(approx) > max_points:
                # 进一步简化
                epsilon = 0.02 * cv2.arcLength(main_contour, True)
                approx = cv2.approxPolyDP(main_contour, epsilon, True)
            
            # 获取边界框
            x_min, y_min, w, h = cv2.boundingRect(main_contour)
            
            # 创建建筑物实例
            building = {
                "id": self.current_building_id,
                "points": [(int(p[0][0]), int(p[0][1])) for p in approx],
                "bbox": [x_min, y_min, x_min + w, y_min + h],
                "mask": best_mask,
                "score": float(best_score),
                "area": int(area),
                "selected": False,
                "contour": main_contour,  # 存储原始轮廓用于像素级操作
                "vertices_count": len(approx)  # 存储顶点数量
            }
            
            self.buildings.append(building)
            self.current_building_id += 1
            
            print(f"✓ 建筑物{self.current_building_id} - 面积: {area:.0f}px², 顶点: {len(building['points'])} (精细度: {self.precision_level})")
            return building
            
        except Exception as e:
            print(f"分割失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _select_best_mask(self, masks, scores):
        """选择最佳掩码"""
        # 优先选择中等面积的掩码（避免过小或过大）
        best_score = -1
        best_idx = 0
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            area = np.sum(mask)
            # 综合得分：置信度 + 面积因子
            area_factor = min(1.0, area / 10000)  # 限制最大影响
            combined_score = score * (1 + 0.2 * area_factor)
            
            if combined_score > best_score and area > 50:
                best_score = combined_score
                best_idx = i
        
        return best_idx
    
    def _get_pixel_precision_contour(self, mask):
        """获取像素级精度的轮廓"""
        # 转换为uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 使用精细的形态学操作
        kernel = np.ones((1, 1), np.uint8)  # 1x1核，最小影响
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # 找到所有轮廓（使用最精确的方法）
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE  # 保留所有点，不压缩
        )
        
        return contours
    
    def update_building_points(self, building_id, new_points):
        """更新建筑物的多边形点（像素级）"""
        for building in self.buildings:
            if building["id"] == building_id:
                # 确保点是整数（像素坐标）
                building["points"] = [(int(x), int(y)) for x, y in new_points]
                
                # 重新计算边界框
                points_array = np.array(building["points"])
                if len(points_array) > 0:
                    x1, y1 = points_array.min(axis=0)
                    x2, y2 = points_array.max(axis=0)
                    building["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
                return True
        return False
    
    def delete_building(self, building_id):
        """删除建筑物"""
        self.buildings = [b for b in self.buildings if b["id"] != building_id]
        print(f"已删除建筑物 {building_id}")
    
    def clear_all_buildings(self):
        """清除所有建筑物"""
        self.buildings.clear()
        print("已清除所有建筑物")
    
    def select_building_at_point(self, x, y):
        """选择包含指定点的建筑物"""
        for building in self.buildings:
            points = np.array(building["points"], dtype=np.int32)
            if cv2.pointPolygonTest(points, (float(x), float(y)), False) >= 0:
                # 清除其他建筑物的选中状态
                for b in self.buildings:
                    b["selected"] = False
                building["selected"] = True
                return building["id"]
        return None
    
    def export_binary_mask(self, output_path=None):
        """导出像素级精度的二值图"""
        if self.image is None:
            print("错误: 未加载图像")
            return None
        
        h, w = self.image.shape[:2]
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 使用fillPoly绘制每个建筑物
        for building in self.buildings:
            points = np.array(building["points"], dtype=np.int32)
            if len(points) > 2:  # 至少需要3个点形成多边形
                # 裁剪坐标到图像范围内
                points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                cv2.fillPoly(binary_mask, [points], 255)
        
        # 将单通道灰度图转换为RGB三通道图像
        binary_mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
        
        # 设置输出路径和格式
        if output_path is None:
            # 使用PNG格式（无损压缩）
            base_name = Path(self.image_path).stem
            output_path = f"{base_name}_buildings.png"
        else:
            # 强制使用PNG格式（无损压缩）
            if not output_path.lower().endswith('.png'):
                output_path = output_path + '.png'
        
        # 使用PNG格式保存，采用Photoshop的压缩算法
        # 1. 差分预测过滤（自动选择最优类型）
        # 2. DEFLATE压缩（LZ77 + 哈夫曼编码，压缩级别6）
        # 3. 元数据清理（仅保留必要信息）
        cv2.imwrite(output_path, binary_mask_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        
        print(f"✓ 已导出二值图: {output_path}, 包含 {len(self.buildings)} 个建筑物")
        return binary_mask

class VertexItem(QGraphicsEllipseItem):
    """顶点控制点 - 1像素大小，悬停变色不变大"""
    
    def __init__(self, x, y, vertex_index, parent_polygon):
        # 1x1像素的顶点
        super().__init__(-0.5, -0.5, 1, 1, parent_polygon)
        self.vertex_index = vertex_index
        self.parent_polygon = parent_polygon
        self.dragging = False
        self.last_scene_pos = None
        
        # 初始位置
        self.setPos(x, y)
        
        # 设置样式（黑白风格）
        self.setBrush(QBrush(QColor(0, 0, 0)))  # 黑色填充
        self.setPen(QPen(QColor(255, 255, 255), 0))  # 无边框
        self.setCursor(QCursor(Qt.CrossCursor))  # 十字光标
        
        # 设置标志
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setZValue(10)  # 确保顶点在最上层
        
        # 设置工具提示
        self.setToolTip(f"顶点 {vertex_index}")
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_scene_pos = event.scenePos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            event.accept()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.dragging and self.last_scene_pos:
            # 获取当前场景位置
            current_scene_pos = event.scenePos()
            
            # 确保在图像范围内
            scene = self.scene()
            if scene and scene.views():
                view = scene.views()[0]
                if hasattr(view, 'image_size'):
                    # 限制在图像范围内
                    x = max(0, min(current_scene_pos.x(), view.image_size.width() - 1))
                    y = max(0, min(current_scene_pos.y(), view.image_size.height() - 1))
                    current_scene_pos = QPointF(x, y)
            
            # 更新顶点位置
            self.setPos(self.parent_polygon.mapFromScene(current_scene_pos))
            
            # 更新父多边形的点
            self.parent_polygon.update_point(self.vertex_index, self.pos())
            
            # 保存当前鼠标位置
            self.last_scene_pos = current_scene_pos
            
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.last_scene_pos = None
            self.setCursor(QCursor(Qt.CrossCursor))
            
            # 通知视图多边形已修改
            if self.parent_polygon.scene() and self.parent_polygon.scene().views():
                view = self.parent_polygon.scene().views()[0]
                if hasattr(view, 'polygon_modified'):
                    view.polygon_modified.emit(self.parent_polygon.building_id, 
                                             self.parent_polygon.get_points())
            
            event.accept()
    
    def itemChange(self, change, value):
        """处理项的变化"""
        if change == QGraphicsItem.ItemPositionHasChanged and self.dragging:
            # 手动调用mouseMoveEvent来处理位置变化
            return value
        
        return super().itemChange(change, value)
    
    def hoverEnterEvent(self, event):
        """悬停进入事件 - 只改变颜色，不改变大小"""
        self.setBrush(QBrush(QColor(255, 255, 255)))  # 悬停时白色
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """悬停离开事件"""
        self.setBrush(QBrush(QColor(0, 0, 0)))  # 黑色
        super().hoverLeaveEvent(event)

class EditablePolygonItem(QGraphicsPolygonItem):
    """可编辑的多边形项 - 1像素边框"""
    
    def __init__(self, points, building_id, parent=None):
        super().__init__(parent)
        self.building_id = building_id
        self.original_points = points.copy()
        self.points = [QPointF(x, y) for x, y in points]
        
        # 更新多边形
        self.update_polygon()
        
        # 设置样式（黑白风格）- 1像素黑色线
        self.setPen(QPen(QColor(0, 0, 0), 1))  # 1像素黑色线
        self.setBrush(QBrush(QColor(0, 0, 0, 30)))  # 轻微半透明黑色
        
        # 顶点项
        self.vertex_items = []
        
        # 状态
        self.selected = False
        
        # 创建顶点（默认隐藏）
        self.create_vertices()
        self.set_selected(False)
        
        # 设置标志
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setCursor(QCursor(Qt.CrossCursor))  # 始终使用十字光标
        self.setZValue(1)
    
    def update_polygon(self):
        """更新多边形形状"""
        if len(self.points) >= 3:  # 至少需要3个点形成多边形
            polygon = QPolygonF(self.points)
            self.setPolygon(polygon)
    
    def create_vertices(self):
        """创建顶点控制点"""
        # 清除现有顶点
        for item in self.vertex_items:
            if item.scene():
                item.scene().removeItem(item)
        self.vertex_items.clear()
        
        # 创建新的顶点
        for i, point in enumerate(self.points):
            vertex = VertexItem(point.x(), point.y(), i, self)
            self.vertex_items.append(vertex)
    
    def set_selected(self, selected):
        """设置选中状态"""
        self.selected = selected
        
        if selected:
            self.setPen(QPen(QColor(0, 0, 0), 1))  # 选中时保持1像素黑色边框
            # 显示顶点
            for vertex in self.vertex_items:
                vertex.show()
                vertex.setBrush(QBrush(QColor(0, 0, 0)))  # 选中时黑色
        else:
            self.setPen(QPen(QColor(0, 0, 0), 1))  # 1像素黑色边框
            # 隐藏顶点
            for vertex in self.vertex_items:
                vertex.hide()
                vertex.setBrush(QBrush(QColor(0, 0, 0)))  # 黑色
    
    def update_point(self, vertex_index, new_pos):
        """更新指定顶点 - 修复坐标转换"""
        if 0 <= vertex_index < len(self.points):
            # 确保点在图像范围内
            scene = self.scene()
            if scene:
                view = scene.views()[0]
                if hasattr(view, 'image_size'):
                    # 转换到场景坐标
                    scene_pos = self.mapToScene(new_pos)
                    # 限制在图像范围内
                    x = max(0, min(scene_pos.x(), view.image_size.width() - 1))
                    y = max(0, min(scene_pos.y(), view.image_size.height() - 1))
                    scene_pos = QPointF(x, y)
                    # 转换回项坐标
                    new_pos = self.mapFromScene(scene_pos)
            
            # 更新点
            self.points[vertex_index] = new_pos
            
            # 更新多边形形状
            self.update_polygon()
            
            # 更新对应顶点项的位置
            if vertex_index < len(self.vertex_items):
                self.vertex_items[vertex_index].setPos(new_pos)
    
    def get_points(self):
        """获取当前点列表"""
        return [(p.x(), p.y()) for p in self.points]
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        # 点击多边形时选择它
        if event.button() == Qt.LeftButton:
            self.set_selected(True)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def hoverEnterEvent(self, event):
        """悬停进入事件"""
        if not self.selected:
            self.setPen(QPen(QColor(50, 50, 50), 1))  # 悬停时深灰色边框
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """悬停离开事件"""
        if not self.selected:
            self.setPen(QPen(QColor(0, 0, 0), 1))  # 黑色边框
        super().hoverLeaveEvent(event)

class BuildingAnnotationView(QGraphicsView):
    """建筑物标注视图 - 支持中键拖动图像"""
    
    # 信号定义
    point_clicked = pyqtSignal(int, int)  # x, y
    building_selected = pyqtSignal(int)  # building_id
    polygon_modified = pyqtSignal(int, list)  # building_id, points
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        # 图像
        self.image_item = None
        self.image_pixmap = None
        self.image_size = QSize()
        
        # 建筑物多边形
        self.building_items = {}  # building_id -> EditablePolygonItem
        self.selected_building = None
        
        # 画笔模式
        self.brush_mode = False
        self.brush_path = QPainterPath()
        self.brush_points = []
        self.brush_item = None
        self.brush_preview_item = None
        
        # 模式
        self.mode = "segment"  # "segment", "edit"
        
        # 交互设置
        self.setDragMode(QGraphicsView.NoDrag)  # 不使用拖动手掌
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # 设置背景（白色）
        self.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        
        # 缩放级别
        self.zoom_level = 1.0
        self.min_zoom = 0.01  # 可以保持较小的最小值
        # 移除 max_zoom 限制或设置一个很大的值
        self.max_zoom = 1000.0  # 设置一个足够大的值
        
        # 设置鼠标跟踪和光标
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.CrossCursor))  # 始终使用十字光标
        
        # 缩放控制
        self.zoom_factor = 1.1
        
        # 中键拖动相关
        self.middle_button_pressed = False
        self.last_mouse_pos = None
        
        # 设置尺寸策略
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def load_image(self, image_array):
        """加载图像"""
        self.scene.clear()
        self.building_items.clear()
        self.selected_building = None
        self.brush_points = []
        self.brush_path = QPainterPath()
        self.brush_item = None
        self.brush_preview_item = None
        
        # 转换图像
        h, w, c = image_array.shape
        bytes_per_line = w * 3
        qimage = QImage(image_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_pixmap = QPixmap.fromImage(qimage)
        
        # 创建图像项
        self.image_item = QGraphicsPixmapItem(self.image_pixmap)
        self.scene.addItem(self.image_item)
        
        # 设置场景大小 - 比图像大，允许图像自由拖动
        scene_margin = max(w, h)  # 添加与图像同尺寸的边距
        self.scene.setSceneRect(-scene_margin, -scene_margin, w + 2 * scene_margin, h + 2 * scene_margin)
        self.image_size = QSize(w, h)
        
        # 调整视图
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.zoom_level = 1.0
        
        # 重置变换
        self.resetTransform()
        self.scale(1.0, 1.0)
    
    def set_mode(self, mode):
        """设置模式"""
        old_mode = self.mode
        self.mode = mode
        
        if mode == "segment":
            # 隐藏所有选择状态
            for item in self.building_items.values():
                item.set_selected(False)
            self.selected_building = None
            
        elif mode == "edit":
            # 如果已有选中的建筑物，显示其顶点
            if self.selected_building in self.building_items:
                self.building_items[self.selected_building].set_selected(True)
    
    def set_brush_mode(self, enabled):
        """设置画笔模式"""
        self.brush_mode = enabled
        if enabled:
            self.brush_points = []
            self.brush_path = QPainterPath()
        else:
            # 清除画笔预览
            if self.brush_item:
                self.scene.removeItem(self.brush_item)
                self.brush_item = None
            if self.brush_preview_item:
                self.scene.removeItem(self.brush_preview_item)
                self.brush_preview_item = None
    
    def add_building(self, building_data):
        """添加建筑物多边形"""
        building_id = building_data["id"]
        
        # 创建多边形项
        polygon_item = EditablePolygonItem(building_data["points"], building_id)
        self.scene.addItem(polygon_item)
        
        # 存储引用
        self.building_items[building_id] = polygon_item
        
        return polygon_item
    
    def add_manual_polygon(self, points):
        """添加手动绘制的多边形"""
        # 生成一个临时的ID（负数表示手动绘制）
        building_id = -len(self.building_items) - 1
        
        # 创建多边形项
        polygon_item = EditablePolygonItem(points, building_id)
        polygon_item.setPen(QPen(QColor(0, 0, 0), 1))  # 1像素黑色，表示手动绘制
        polygon_item.setBrush(QBrush(QColor(0, 0, 0, 30)))
        self.scene.addItem(polygon_item)
        
        # 存储引用
        self.building_items[building_id] = polygon_item
        
        return building_id
    
    def select_building(self, building_id):
        """选择建筑物"""
        # 清除之前的选择
        if self.selected_building in self.building_items:
            self.building_items[self.selected_building].set_selected(False)
        
        # 设置新的选择
        if building_id in self.building_items:
            self.building_items[building_id].set_selected(True)
            self.selected_building = building_id
            self.building_selected.emit(building_id)
    
    def delete_building(self, building_id):
        """删除建筑物"""
        if building_id in self.building_items:
            item = self.building_items.pop(building_id)
            self.scene.removeItem(item)
            
            # 如果删除的是当前选中的建筑物，重置选中状态
            if self.selected_building == building_id:
                self.selected_building = None
    
    def clear_all_buildings(self):
        """清除所有建筑物"""
        for item in self.building_items.values():
            self.scene.removeItem(item)
        self.building_items.clear()
        self.selected_building = None
        
    def generate_building_items(self, buildings):
        """从建筑物列表生成建筑物图形项"""
        # 清除现有所有建筑物图形项
        for item in self.building_items.values():
            self.scene.removeItem(item)
        self.building_items.clear()
        self.selected_building = None
        
        # 为每个建筑物创建图形项
        for building in buildings:
            self.add_building(building)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.image_item is None:
            super().mousePressEvent(event)
            return
        
        scene_pos = self.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        
        # 检查是否点击了多边形
        items = self.scene.items(scene_pos)
        for item in items:
            if isinstance(item, EditablePolygonItem):
                self.select_building(item.building_id)
                return
        
        # 如果没有点击多边形，继续处理其他事件
        # 检查是否在图像范围内 - 只对非画笔模式应用限制
        if not self.brush_mode and (x < 0 or y < 0 or x > self.image_size.width() or y > self.image_size.height()):
            super().mousePressEvent(event)
            return
        
        if event.button() == Qt.LeftButton:
            if self.brush_mode:
                # 画笔模式：添加点 - 1像素笔刷
                self.brush_points.append((x, y))
                
                # 更新画笔路径
                self.update_brush_path()
                
                # 不自动闭合多边形，等待用户手动连接起点
                event.accept()
                return
                
            elif self.mode == "segment":
                # 分割模式：点击进行建筑物分割
                self.point_clicked.emit(int(x), int(y))
                event.accept()
                return
                
            elif self.mode == "edit":
                # 首先传递事件给场景，让顶点项有机会处理
                super().mousePressEvent(event)
                
                # 如果事件没有被接受，检查是否点击了建筑物
                if not event.isAccepted():
                    # 检查是否点击了建筑物
                    building_clicked = False
                    
                    for building_id, item in self.building_items.items():
                        if item.contains(scene_pos):
                            self.select_building(building_id)
                            building_clicked = True
                            break
                    
                    if not building_clicked:
                        # 点击在空白处，清除选择
                        if self.selected_building in self.building_items:
                            self.building_items[self.selected_building].set_selected(False)
                        self.selected_building = None
                        self.building_selected.emit(-1)
                
                return
        
        elif event.button() == Qt.RightButton:
            if self.brush_mode and self.brush_points:
                # 右键完成画笔绘制
                self.finish_brush_polygon()
                event.accept()
                return
            elif self.mode == "edit":
                # 右键删除选中的建筑物
                if self.selected_building is not None:
                    self.delete_building(self.selected_building)
                    self.selected_building = None
                    event.accept()
                    return
                else:
                    # 在编辑模式下，在多边形外部右键点击，清除所有选择
                    for item in self.building_items.values():
                        item.set_selected(False)
                    self.selected_building = None
                    self.building_selected.emit(-1)
                    event.accept()
                    return
        
        elif event.button() == Qt.MiddleButton:
            # 中键按下，开始拖动
            self.middle_button_pressed = True
            self.last_mouse_pos = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            event.accept()
            return
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        # 处理中键拖动
        if self.middle_button_pressed and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            
            # 平移视图
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        
        scene_pos = self.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        
        if self.brush_mode and self.brush_points:
            # 实时预览画笔路径 - 1像素笔刷
            if len(self.brush_points) > 0:
                # 移除旧的预览项
                if self.brush_preview_item:
                    self.scene.removeItem(self.brush_preview_item)
                
                # 创建预览路径
                preview_path = QPainterPath()
                first_x, first_y = self.brush_points[0]
                preview_path.moveTo(first_x, first_y)
                
                for px, py in self.brush_points[1:]:
                    preview_path.lineTo(px, py)
                
                # 连接到当前鼠标位置
                preview_path.lineTo(x, y)
                
                # 仅在鼠标靠近起点时，提供辅助连线到起点
                if len(self.brush_points) > 1:
                    distance = ((x - first_x)**2 + (y - first_y)**2)**0.5
                    if distance < 15:  # 当鼠标靠近起点时，显示辅助连线
                        preview_path.lineTo(first_x, first_y)
                
                # 创建预览项
                self.brush_preview_item = QGraphicsPathItem(preview_path)
                self.brush_preview_item.setPen(QPen(QColor(0, 0, 0, 150), 0.5, Qt.DashLine))  # 0.5像素虚线
                self.scene.addItem(self.brush_preview_item)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MiddleButton:
            self.middle_button_pressed = False
            self.last_mouse_pos = None
            self.setCursor(QCursor(Qt.CrossCursor))
            event.accept()
            return
        
        super().mouseReleaseEvent(event)
    
    def update_brush_path(self):
        """更新画笔路径 - 1像素笔刷"""
        if not self.brush_points:
            return
        
        # 移除旧的画笔项
        if self.brush_item:
            self.scene.removeItem(self.brush_item)
        
        # 创建新的路径
        path = QPainterPath()
        first_x, first_y = self.brush_points[0]
        path.moveTo(first_x, first_y)
        
        for x, y in self.brush_points[1:]:
            path.lineTo(x, y)
        
        # 绘制当前点
        if len(self.brush_points) > 0:
            last_x, last_y = self.brush_points[-1]
            path.addEllipse(last_x - 0.5, last_y - 0.5, 1, 1)  # 1像素点
        
        # 创建路径项
        self.brush_item = QGraphicsPathItem(path)
        self.brush_item.setPen(QPen(QColor(0, 0, 0), 0.5))  # 0.5像素黑线
        self.scene.addItem(self.brush_item)
    
    def finish_brush_polygon(self):
        """完成画笔多边形"""
        if len(self.brush_points) < 3:
            return
        
        # 确保多边形闭合
        if self.brush_points[0] != self.brush_points[-1]:
            self.brush_points.append(self.brush_points[0])
        
        # 创建多边形项
        building_id = self.add_manual_polygon(self.brush_points)
        
        # 发射信号
        self.polygon_modified.emit(building_id, self.brush_points)
        
        # 重置画笔
        self.brush_points = []
        if self.brush_item:
            self.scene.removeItem(self.brush_item)
            self.brush_item = None
        if self.brush_preview_item:
            self.scene.removeItem(self.brush_preview_item)
            self.brush_preview_item = None
    
    def wheelEvent(self, event):
        """滚轮缩放 - 移除上限限制"""
        # 计算缩放因子
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = self.zoom_factor if zoom_in else 1/self.zoom_factor
        
        # 只保留最小缩放限制，移除最大缩放限制
        new_zoom = self.zoom_level * zoom_factor
        if new_zoom < self.min_zoom:  # 只检查最小限制
            return
        
        # 缩放（锚点已在__init__中设置为AnchorUnderMouse）
        self.scale(zoom_factor, zoom_factor)
        self.zoom_level = new_zoom
    
    def fit_to_view(self):
        """适应视图"""
        if self.image_item:
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.zoom_level = 1.0
    
    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key_Delete and self.selected_building is not None:
            # 删除选中的建筑物
            self.delete_building(self.selected_building)
            self.selected_building = None
        elif event.key() == Qt.Key_F:
            self.fit_to_view()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # 放大
            if self.zoom_level * 1.1 <= self.max_zoom:
                self.scale(1.1, 1.1)
                self.zoom_level *= 1.1
        elif event.key() == Qt.Key_Minus:
            # 缩小
            if self.zoom_level / 1.1 >= self.min_zoom:
                self.scale(1/1.1, 1/1.1)
                self.zoom_level /= 1.1
        elif event.key() == Qt.Key_Escape and self.brush_mode:
            # ESC键退出画笔模式
            self.brush_points = []
            if self.brush_item:
                self.scene.removeItem(self.brush_item)
                self.brush_item = None
            if self.brush_preview_item:
                self.scene.removeItem(self.brush_preview_item)
                self.brush_preview_item = None
        
        super().keyPressEvent(event)

class ImageFileList(QWidget):
    """图像文件列表控件"""
    
    image_selected = pyqtSignal(str)  # 图像路径
    
    def __init__(self, title="图像列表", parent=None):
        super().__init__(parent)
        self.title = title
        self.folder_path = None
        self.image_files = []
        self.image_paths_set = set()
        self.dark_mode = False
        self.current_image_path = None  # 存储当前选中的图片路径
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        # 其余位置字体恢复为较小，保持紧凑
        title_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px; border-bottom: 1px solid #ccc;")
        layout.addWidget(title_label)
        
        # 文件夹信息
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("未选择文件夹")
        self.folder_label.setStyleSheet("color: #666; font-size: 11px;")
        self.folder_label.setWordWrap(True)
        folder_layout.addWidget(self.folder_label)
        
        self.select_btn = QPushButton("📁")
        self.select_btn.setToolTip("选择文件夹")
        self.select_btn.setFixedSize(40, 40)
        self.select_btn.clicked.connect(self.select_folder)
        self.select_btn.setStyleSheet("font-size: 18px; padding: 0;")
        folder_layout.addWidget(self.select_btn)
        
        # 添加单张图片按钮
        self.add_image_btn = QPushButton("+")
        self.add_image_btn.setToolTip("添加单张图片")
        self.add_image_btn.setFixedSize(40, 40)
        self.add_image_btn.clicked.connect(self.add_single_image)
        self.add_image_btn.setStyleSheet("font-size: 24px; font-weight: bold; padding: 0;")
        folder_layout.addWidget(self.add_image_btn)
        
        layout.addLayout(folder_layout)
        
        # 搜索框
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索图片...")
        self.search_input.textChanged.connect(self.filter_images)
        search_layout.addWidget(self.search_input)
        
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setToolTip("刷新")
        self.refresh_btn.setFixedSize(40, 40)
        self.refresh_btn.clicked.connect(self.refresh_list)
        self.refresh_btn.setStyleSheet("font-size: 20px; padding: 0;")
        search_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(search_layout)
        
        # 图片列表
        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self.on_image_double_clicked)
        self.image_list.setAlternatingRowColors(True)
        self.image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.image_list)
        
        # 状态标签
        self.status_label = QLabel("0 个图像")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 11px; border-top: 1px solid #ccc; padding: 5px;")
        layout.addWidget(self.status_label)
    
    def select_folder(self):
        """选择文件夹"""
        folder = QFileDialog.getExistingDirectory(self, f"选择{self.title}文件夹")
        if folder:
            self.folder_path = folder
            self.folder_label.setText(f"📂 {Path(folder).name}")
            self.folder_label.setToolTip(folder)
            self.load_images_from_folder(folder)
    
    def add_single_image(self):
        """添加单张图片"""
        image_files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片文件", "", 
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.JPG *.JPEG *.PNG *.BMP *.TIFF *.TIF)"
        )
        
        if image_files:
            for image_file in image_files:
                self.add_image_path(image_file)
            
            self.update_image_list()
            self.status_label.setText(f"{len(self.image_files)} 个图像")
            
            # 如果有图片，选择第一个
            if self.image_files:
                self.image_list.setCurrentRow(0)
    
    def add_image_path(self, image_path):
        """添加单张图片路径"""
        image_path_str = str(image_path)
        if image_path_str not in self.image_paths_set:
            self.image_paths_set.add(image_path_str)
            self.image_files.append(Path(image_path_str))
    
    def load_images_from_folder(self, folder_path):
        """从文件夹加载图片"""
        self.folder_path = folder_path
        self.image_files = []
        self.image_paths_set.clear()
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # 搜索所有图片文件（不区分大小写）
        for ext in image_extensions:
            # 搜索小写扩展名
            for img_path in Path(folder_path).glob(f"*{ext}"):
                self.add_image_path(img_path)
            
            # 搜索大写扩展名
            for img_path in Path(folder_path).glob(f"*{ext.upper()}"):
                self.add_image_path(img_path)
        
        # 按文件名排序
        self.image_files.sort(key=lambda x: x.name)
        
        # 更新列表
        self.update_image_list()
        
        # 更新状态
        self.status_label.setText(f"{len(self.image_files)} 个图像")
        
        # 如果有图片，选择第一个
        if self.image_files:
            self.image_list.setCurrentRow(0)
    
    # 替换整个 update_image_list 方法
    def update_image_list(self, filter_text="", current_image_path=None):
        """更新图片列表，支持标记当前操作的图片"""
        # 保存当前滚动位置
        scroll_position = self.image_list.verticalScrollBar().value()
        
        self.image_list.clear()
        
        for image_path in self.image_files:
            if filter_text.lower() in image_path.name.lower():
                item = QListWidgetItem(image_path.name)
                item.setData(Qt.UserRole, str(image_path))
                
                # 检查是否是当前操作的图片
                if self.current_image_path and str(image_path) == self.current_image_path:
                    # 使用更加醒目的颜色和加粗字体来突出显示当前选中的图片
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    
                    # 根据主题选择更深的颜色，确保更明显
                    if self.dark_mode:
                        item.setForeground(QColor(255, 255, 255))  # 黑暗模式下的纯白色
                    else:
                        item.setForeground(QColor(0, 0, 0))  # 光明模式下的纯黑色
                else:
                    # 非选中项保持原有颜色
                    if self.dark_mode:
                        item.setForeground(QColor(200, 200, 200))  # 深色模式下的浅灰色
                    else:
                        item.setForeground(QColor(30, 30, 30))  # 光明模式下的深灰色
                
                self.image_list.addItem(item)
        
        # 恢复滚动位置
        self.image_list.verticalScrollBar().setValue(scroll_position)
        
        # 更新状态
        self.status_label.setText(f"{self.image_list.count()} 个图像 (共 {len(self.image_files)} 个)")
    
    def filter_images(self, text):
        """过滤图片"""
        self.update_image_list(text)
    
    # 替换整个 mark_current_image 方法
    def mark_current_image(self, current_image_path):
        """标记当前操作的图片"""
        self.current_image_path = current_image_path
        # 传递当前的搜索文本，保持筛选状态
        self.update_image_list(self.search_input.text())
    
    def set_dark_mode(self, dark_mode):
        """设置深色模式"""
        self.dark_mode = dark_mode
        # 重新更新列表以应用新颜色
        self.update_image_list(self.search_input.text())
    
    def refresh_list(self):
        """刷新列表"""
        if self.folder_path:
            self.load_images_from_folder(self.folder_path)
    
    def on_image_double_clicked(self, item):
        """图片双击事件"""
        image_path = item.data(Qt.UserRole)
        self.image_selected.emit(image_path)
    
    def show_context_menu(self, pos):
        """显示右键菜单"""
        menu = QMenu()
        remove_action = QAction("从列表中移除", self)
        remove_action.triggered.connect(self.remove_selected_image)
        menu.addAction(remove_action)
        
        clear_action = QAction("清空列表", self)
        clear_action.triggered.connect(self.clear_list)
        menu.addAction(clear_action)
        
        menu.exec_(self.image_list.mapToGlobal(pos))
    
    def remove_selected_image(self):
        """从列表中移除选中的图片"""
        current_item = self.image_list.currentItem()
        if current_item:
            current_row = self.image_list.currentRow()
            image_path = current_item.data(Qt.UserRole)
            
            # 从数据中移除
            for i, img_path in enumerate(self.image_files):
                if str(img_path) == image_path:
                    self.image_files.pop(i)
                    self.image_paths_set.remove(image_path)
                    break
            
            # 更新列表
            self.update_image_list(self.search_input.text())
            
            # 如果还有图片，选择下一个或上一个
            if self.image_list.count() > 0:
                if current_row >= self.image_list.count():
                    self.image_list.setCurrentRow(self.image_list.count() - 1)
                else:
                    self.image_list.setCurrentRow(current_row)
    
    def clear_list(self):
        """清空列表"""
        self.image_files = []
        self.image_paths_set.clear()
        self.image_list.clear()
        self.status_label.setText("0 个图像")
        self.folder_label.setText("未选择文件夹")
        self.folder_path = None
    
    def get_current_image(self):
        """获取当前选中的图片"""
        current_item = self.image_list.currentItem()
        if current_item:
            return current_item.data(Qt.UserRole)
        return None
    
    def get_next_image(self):
        """获取下一张图片"""
        current_row = self.image_list.currentRow()
        if current_row < self.image_list.count() - 1:
            self.image_list.setCurrentRow(current_row + 1)
            next_image = self.get_current_image()
            if next_image:
                self.mark_current_image(next_image)
            return next_image
        return None
    
    def get_previous_image(self):
        """获取上一张图片"""
        current_row = self.image_list.currentRow()
        if current_row > 0:
            self.image_list.setCurrentRow(current_row - 1)
            prev_image = self.get_current_image()
            if prev_image:
                self.mark_current_image(prev_image)
            return prev_image
        return None

class FreeAnnotationDialog(QDialog):
    """自由标注对话框 - 弹出式标注窗口"""
    
    # 信号：标注同步
    building_added = pyqtSignal(dict, str)  # 建筑物数据, 图像类型
    building_deleted = pyqtSignal(int, str)  # 建筑物ID, 图像类型
    polygon_updated = pyqtSignal(int, list, str)  # 建筑物ID, 点列表, 图像类型
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sam_instance = None
        self.current_image = None
        self.undo_stack = []
        self.max_undo_steps = 20
        self.image_type = ""  # "before" 或 "after"
        self.fullscreen_mode = False
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('自由标注模式')
        self.setGeometry(100, 100, 1400, 900)  # 初始窗口更大
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 顶部控制栏
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # 图像选择
        self.image_label = QLabel("未选择图像")
        top_layout.addWidget(self.image_label)
        
        top_layout.addStretch()
        
        # 工具模式
        self.tool_group = QButtonGroup(self)
        
        self.sam_mode_btn = QRadioButton("🔍 SAM分割")
        self.sam_mode_btn.setChecked(True)
        self.sam_mode_btn.toggled.connect(lambda: self.set_tool_mode("sam"))
        top_layout.addWidget(self.sam_mode_btn)
        
        self.brush_mode_btn = QRadioButton("✏️ 画笔")
        self.brush_mode_btn.toggled.connect(lambda: self.set_tool_mode("brush"))
        top_layout.addWidget(self.brush_mode_btn)
        
        self.edit_mode_btn = QRadioButton("📝 编辑")
        self.edit_mode_btn.toggled.connect(lambda: self.set_tool_mode("edit"))
        top_layout.addWidget(self.edit_mode_btn)
        
        top_layout.addStretch()
        
        # 新增：精细度显示和调整
        precision_layout = QHBoxLayout()
        precision_layout.addWidget(QLabel("精细度:"))
        
        self.precision_value_label = QLabel("5")
        self.precision_value_label.setFixedWidth(20)
        precision_layout.addWidget(self.precision_value_label)
        
        self.precision_slider = QSlider(Qt.Horizontal)
        self.precision_slider.setRange(1, 10)
        self.precision_slider.setValue(5)
        self.precision_slider.setFixedWidth(80)
        self.precision_slider.valueChanged.connect(self.update_precision)
        precision_layout.addWidget(self.precision_slider)
        
        top_layout.addLayout(precision_layout)
        
        top_layout.addStretch()
        
        # 操作按钮
        self.fullscreen_btn = QPushButton("⛶ 全屏")
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_btn.setToolTip("切换全屏模式 (F11)")
        top_layout.addWidget(self.fullscreen_btn)
        
        self.undo_btn = QPushButton("↶ 撤销")
        self.undo_btn.clicked.connect(self.undo_last_action)
        self.undo_btn.setEnabled(False)
        top_layout.addWidget(self.undo_btn)
        
        self.clear_btn = QPushButton("🗑️ 清除")
        self.clear_btn.clicked.connect(self.clear_all_annotations)
        top_layout.addWidget(self.clear_btn)
        
        self.export_btn = QPushButton("💾 导出")
        self.export_btn.clicked.connect(self.export_binary_mask)
        self.export_btn.setEnabled(False)
        top_layout.addWidget(self.export_btn)
        
        self.close_btn = QPushButton("✕ 关闭")
        self.close_btn.clicked.connect(self.close)
        top_layout.addWidget(self.close_btn)
        
        main_layout.addWidget(top_bar)
        
        # 图像视图
        self.view = BuildingAnnotationView()
        self.view.point_clicked.connect(self.on_point_clicked)
        self.view.building_selected.connect(self.on_building_selected)
        self.view.polygon_modified.connect(self.on_polygon_modified)
        main_layout.addWidget(self.view)
        
        # 底部状态栏
        bottom_bar = QWidget()
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.status_label = QLabel("就绪 - 点击图像进行标注 | 滚轮缩放 | +/-键缩放 | F11全屏")
        bottom_layout.addWidget(self.status_label)
        
        bottom_layout.addStretch()
        
        self.building_info = QLabel("建筑物: 0")
        bottom_layout.addWidget(self.building_info)
        
        main_layout.addWidget(bottom_bar)
    
    def update_precision(self, value):
        """更新多边形精细度"""
        self.precision_value_label.setText(str(value))
        
        if self.sam_instance:
            self.sam_instance.set_precision_level(value)
            self.status_label.setText(f"多边形精细度设置为: {value}")
    
    def set_image(self, image_path, sam_predictor, image_type="before"):
        """设置图像和SAM预测器"""
        self.image_type = image_type
        
        try:
            # 显示加载状态
            self.status_label.setText("正在加载图像...")
            QApplication.processEvents()
            
            # 检查缓存
            cached_image = ImageCache.get(image_path)
            if cached_image is not None:
                print(f"自由标注模式: 从缓存加载图像: {image_path}")
                image_array = cached_image
            else:
                # 支持中文路径的图像加载方法
                def imread_chinese(path, mode=cv2.IMREAD_COLOR):
                    """读取带中文路径的图像"""
                    import numpy as np
                    with open(path, 'rb') as f:
                        image_data = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(image_data, mode)
                    return img
                
                # 快速加载图像
                img = imread_chinese(image_path)
                if img is None:
                    raise ValueError(f"无法读取图像: {image_path}")
                image_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 缓存图像
                ImageCache.set(image_path, image_array)
                print(f"自由标注模式: 从磁盘加载图像: {image_path}")
            
            # 创建SAM实例（共享预测器，但不复制其他状态）
            self.sam_instance = PixelPrecisionSAM()
            self.sam_instance.predictor = sam_predictor
            self.sam_instance.image = image_array
            self.sam_instance.image_path = image_path
            
            # 设置图像到预测器
            self.sam_instance.predictor.set_image(image_array)
            
            # 设置精细度（从主窗口获取或使用默认值）
            self.sam_instance.set_precision_level(self.precision_slider.value())
            
            # 加载到视图
            self.view.load_image(image_array)
            self.current_image = image_array
            
            # 更新UI
            self.image_label.setText(f"📷 {Path(image_path).name} ({image_type})")
            self.export_btn.setEnabled(True)
            self.update_building_count()
            
            # 清空撤销栈
            self.undo_stack.clear()
            self.update_undo_button_state()
            
            self.status_label.setText(f"已加载图像: {Path(image_path).name} - 标注会自动同步到主窗口")
            
        except Exception as e:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("错误")
            msg_box.setText(f"加载图像失败:\n{str(e)}")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
            msg_box.exec_()
    
    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.fullscreen_mode:
            self.showNormal()
            self.fullscreen_btn.setText("⛶ 全屏")
            self.fullscreen_mode = False
        else:
            self.showFullScreen()
            self.fullscreen_btn.setText("⛶ 窗口")
            self.fullscreen_mode = True
    
    def set_tool_mode(self, mode):
        """设置工具模式"""
        if mode == "sam":
            self.view.set_mode("segment")
            self.view.set_brush_mode(False)
            self.status_label.setText("SAM分割模式：点击建筑物进行自动分割")
        elif mode == "brush":
            self.view.set_brush_mode(True)
            self.view.set_mode("segment")
            self.status_label.setText("画笔绘制模式：左键添加顶点，右键完成绘制")
        elif mode == "edit":
            self.view.set_mode("edit")
            self.view.set_brush_mode(False)
            self.status_label.setText("编辑模式：选择建筑物，拖动顶点进行微调")
    
    def save_state(self):
        """保存当前状态"""
        if self.sam_instance is None:
            return
        
        # 获取当前SAM建筑物数据
        sam_buildings = copy.deepcopy(self.sam_instance.buildings) if self.sam_instance else []
        
        # 获取当前视图中的手动建筑物数据
        manual_buildings = []
        for building_id, item in self.view.building_items.items():
            if building_id < 0:  # 手动绘制的建筑物
                manual_buildings.append({
                    'id': building_id,
                    'points': item.get_points()
                })
        
        # 创建状态对象
        state = {
            'sam_buildings': sam_buildings,
            'manual_buildings': manual_buildings,
            'current_building_id': self.sam_instance.current_building_id if self.sam_instance else 0
        }
        
        # 添加到撤销栈
        self.undo_stack.append(state)
        
        # 限制栈大小
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
        
        # 更新撤销按钮状态
        self.update_undo_button_state()
        
        # 更新建筑物计数
        self.update_building_count()
    
    def undo_last_action(self):
        """撤销最近的操作"""
        if not self.undo_stack:
            return
        
        # 获取最近的状态
        state = self.undo_stack.pop()
        
        # 恢复SAM建筑物
        if self.sam_instance:
            self.sam_instance.buildings = copy.deepcopy(state['sam_buildings'])
            self.sam_instance.current_building_id = state['current_building_id']
        
        # 清除视图中的所有建筑物
        self.view.clear_all_buildings()
        
        # 恢复SAM建筑物到视图
        for building in state['sam_buildings']:
            self.view.add_building(building)
        
        # 恢复手动建筑物到视图
        for building in state['manual_buildings']:
            self.view.add_manual_polygon(building['points'])
        
        # 更新撤销按钮状态
        self.update_undo_button_state()
        
        # 更新建筑物计数
        self.update_building_count()
        
        # 同步到主窗口
        self.sync_to_main_window()
        
        self.status_label.setText("已撤销最近操作")
    
    def update_undo_button_state(self):
        """更新撤销按钮状态"""
        self.undo_btn.setEnabled(len(self.undo_stack) > 0)
    
    def update_building_count(self):
        """更新建筑物计数"""
        sam_count = len(self.sam_instance.buildings) if self.sam_instance else 0
        manual_count = len([bid for bid in self.view.building_items.keys() if bid < 0])
        self.building_info.setText(f"建筑物: {sam_count + manual_count}")
    
    def on_point_clicked(self, x, y):
        """处理点击事件"""
        if self.sam_instance is None or self.current_image is None:
            return
        
        # 保存当前状态
        self.save_state()
        
        # 显示处理状态
        self.status_label.setText(f"正在分割建筑物 ({x}, {y})...")
        QApplication.processEvents()
        
        # 调用SAM分割建筑物
        building = self.sam_instance.segment_building(x, y)
        
        if building:
            # 在视图中添加建筑物
            self.view.add_building(building)
            
            # 同步到主窗口
            self.sync_building_to_main(building)
            
            # 更新建筑物计数
            self.update_building_count()
            
            self.status_label.setText(f"✅ 已分割建筑物 {building['id']} - 面积: {building['area']}px²")
        else:
            self.status_label.setText("❌ 分割失败，请尝试点击建筑物的不同位置")
    
    def on_building_selected(self, building_id):
        """建筑物被选中"""
        pass  # 在自由标注模式中，不需要特殊处理
    
    def on_polygon_modified(self, building_id, points):
        """多边形修改事件"""
        # 在修改完成后保存状态
        self.save_state()
        
        # 同步到主窗口
        self.polygon_updated.emit(building_id, points, self.image_type)
    
    def sync_building_to_main(self, building):
        """同步建筑物到主窗口"""
        self.building_added.emit(building, self.image_type)
    
    def sync_to_main_window(self):
        """同步所有建筑物到主窗口"""
        # 清除主窗口的标注
        self.building_deleted.emit(-999, self.image_type)  # 发送特殊ID表示清除所有
        
        # 同步SAM建筑物
        if self.sam_instance:
            for building in self.sam_instance.buildings:
                self.building_added.emit(building, self.image_type)
        
        # 同步手动建筑物
        for building_id, item in self.view.building_items.items():
            if building_id < 0:  # 手动绘制的建筑物
                building_data = {
                    "id": building_id,
                    "points": item.get_points(),
                    "area": cv2.contourArea(np.array(item.get_points(), dtype=np.int32)),
                    "score": 0
                }
                self.building_added.emit(building_data, self.image_type)
    
    def clear_all_annotations(self):
        """清除所有标注"""
        msg_box = QMessageBox()
        msg_box.setWindowTitle("确认")
        msg_box.setText("确定要清除所有标注吗？\n这将同时清除主窗口中的标注")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
        reply = msg_box.exec_()
        
        if reply == QMessageBox.Yes:
            # 在清除前保存状态
            self.save_state()
            
            if self.sam_instance:
                self.sam_instance.clear_all_buildings()
            self.view.clear_all_buildings()
            
            # 同步到主窗口
            self.building_deleted.emit(-999, self.image_type)  # 发送特殊ID表示清除所有
            
            # 更新建筑物计数
            self.update_building_count()
            
            self.status_label.setText("已清除所有建筑物标注")
    
    def export_binary_mask(self):
        """导出二值图"""
        if self.sam_instance is None or self.current_image is None:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("警告")
            msg_box.setText("请先加载图像")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
            msg_box.exec_()
            return
        
        # 计算建筑物数量
        sam_count = len(self.sam_instance.buildings) if self.sam_instance else 0
        manual_count = len([bid for bid in self.view.building_items.keys() if bid < 0])
        
        if sam_count + manual_count == 0:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有可导出的建筑物标注")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
            msg_box.exec_()
            return
        
        # 设置初始保存路径 - 始终使用PNG格式（无损压缩）
        initial_path = self.last_export_folder
        if initial_path and Path(initial_path).is_dir():
            initial_path = Path(initial_path) / f"{Path(self.sam_instance.image_path).stem}_free_buildings.png"
        else:
            initial_path = f"{Path(self.sam_instance.image_path).stem}_free_buildings.png"
        
        # 源图片格式（始终为PNG）
        source_format = '.png'
        
        # 设置文件对话框过滤器 - 只显示PNG格式
        file_filter = "PNG图像 (*.png)"
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存建筑物标签", 
            str(initial_path),  # 将Path对象转换为字符串
            file_filter
        )
        
        if save_path:
            try:
                # 创建一个新的二值图
                h, w = self.current_image.shape[:2]
                binary_mask = np.zeros((h, w), dtype=np.uint8)
                
                # 绘制SAM分割的建筑物
                if self.sam_instance:
                    for building in self.sam_instance.buildings:
                        points = np.array(building["points"], dtype=np.int32)
                        if len(points) > 2:
                            # 裁剪坐标到图像范围内
                            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                            points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                            cv2.fillPoly(binary_mask, [points], 255)
                
                # 绘制手动绘制的建筑物
                for building_id, item in self.view.building_items.items():
                    if building_id < 0:  # 手动绘制的建筑物
                        points = np.array(item.get_points(), dtype=np.int32)
                        if len(points) > 2:
                            # 裁剪坐标到图像范围内
                            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                            points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                            cv2.fillPoly(binary_mask, [points], 255)
                
                # 将单通道灰度图转换为RGB三通道图像
                binary_mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
                
                # 确保保存为PNG格式（无损压缩）
                if save_path.lower().endswith('.png'):
                    final_path = save_path
                else:
                    final_path = save_path + '.png'
                
                # 使用PNG格式保存，采用Photoshop的压缩算法
                # 1. 差分预测过滤（自动选择最优类型）
                # 2. DEFLATE压缩（LZ77 + 哈夫曼编码，压缩级别6）
                # 3. 元数据清理（仅保留必要信息）
                cv2.imwrite(final_path, binary_mask_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                
                msg_box = QMessageBox()
                msg_box.setWindowTitle("成功")
                msg_box.setText(f"✅ 二值图已保存到:\n{final_path}\n"
                    f"建筑物总数: {sam_count + manual_count}\n"
                    f"图像尺寸: {self.current_image.shape[1]}×{self.current_image.shape[0]}\n"
                    f"颜色模式: RGB (3通道)\n"
                    f"文件格式: {source_format[1:].upper()}")
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
                msg_box.exec_()
                # 更新上次导出的文件夹路径
                self.last_export_folder = str(Path(final_path).parent)
                
                self.status_label.setText(f"已导出二值图: {final_path}")
            except Exception as e:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("错误")
                msg_box.setText(f"导出失败:\n{str(e)}")
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
                msg_box.exec_()
    
    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key_F11:
            # F11 全屏
            self.toggle_fullscreen()
        elif event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            # Ctrl+Z 撤销
            self.undo_last_action()
        elif event.key() == Qt.Key_P:
            # P键调整精细度
            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                # 增加精细度
                if self.precision_slider.value() < 10:
                    self.precision_slider.setValue(self.precision_slider.value() + 1)
            elif event.key() == Qt.Key_Minus:
                # 减少精细度
                if self.precision_slider.value() > 1:
                    self.precision_slider.setValue(self.precision_slider.value() - 1)
        
        super().keyPressEvent(event)

class ToolButton(QToolButton):
    """自定义工具按钮，带图标和文字"""
    def __init__(self, icon_text, text, parent=None):
        super().__init__(parent)
        self.setText(f"{icon_text} {text}")
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.setFixedHeight(35)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

class ToolBarDockWidget(QDockWidget):
    """可停靠的工具栏窗口部件"""
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # 创建内容部件
        content_widget = QWidget()
        self.setWidget(content_widget)
        # 注意：不要在这里返回content_widget
    
    # 如果需要访问内部的内容部件，可以通过widget()方法或者添加一个专门的方法
    def contentWidget(self):
        """获取内容部件"""
        return self.widget()

class SAMBuildingAnnotator(QMainWindow):
    """SAM建筑物像素级标注工具 - 遥感变化检测版本"""
    
    def __init__(self):
        super().__init__()
        self.sam = None
        self.current_image_before = None  # 先时序图像
        self.current_image_after = None   # 后时序图像
        self.manual_buildings = []  # 手动绘制的建筑物
        self.last_export_folder = ""  # 存储上次导出的文件夹路径
        
        # 对比模式相关属性
        self.compare_mode = False
        self.after_image_visible = True
        self.current_selected_building = None
        
        # 撤销栈
        self.undo_stack_before = []  # 先时序撤销栈
        self.undo_stack_after = []   # 后时序撤销栈
        self.undo_stack = []         # 统一撤销栈（按时间顺序）
        self.max_undo_steps = 20     # 最大撤销步数
        
        # 图像加载进度
        self.loading_dialog = None
        
        # 自由标注窗口引用
        self.free_annotation_dialogs = []
        
        # 新增：精细度级别 (1-10)
        self.precision_level = 5
        
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('STARbuliding')
        self.setGeometry(50, 50, 1800, 1000)  # 更大的窗口尺寸
        
        # 设置窗口图标
        self.setWindowIcon(QIcon(resource_path('favicon.ico')))
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建主水平分割器（用于左右图像区域）
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        

        
        # 左侧：先时序图像区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        # 左侧文件列表（可折叠）
        self.left_file_dock = ToolBarDockWidget("先时序图像 (时间较早)")
        left_file_content = self.left_file_dock.widget()
        left_file_layout = QVBoxLayout(left_file_content)
        left_file_layout.setContentsMargins(5, 5, 5, 5)
        
        self.left_file_list = ImageFileList("先时序图像")
        left_file_layout.addWidget(self.left_file_list)
        self.left_file_list.image_selected.connect(self.load_before_image)
        
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_file_dock)
        
        # 左侧图像视图
        self.left_view = BuildingAnnotationView()
        self.left_view.point_clicked.connect(self.on_point_clicked_before)
        self.left_view.building_selected.connect(lambda id: self.on_building_selected(id, "before"))
        self.left_view.polygon_modified.connect(lambda id, points: self.on_polygon_modified(id, points, "before"))
        left_layout.addWidget(self.left_view)
        
        # 左侧状态标签
        self.left_status = QLabel("先时序: 未加载图像")
        self.left_status.setStyleSheet("font-size: 11px; padding: 5px; border-top: 1px solid #ccc; background-color: #f8f8f8;")
        left_layout.addWidget(self.left_status)
        
        left_container = QWidget()
        left_container.setLayout(left_layout)
        main_splitter.addWidget(left_container)
        
        # 右侧：后时序图像区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # 右侧文件列表（可折叠）
        self.right_file_dock = ToolBarDockWidget("后时序图像 (时间较晚)")
        right_file_content = self.right_file_dock.widget()
        right_file_layout = QVBoxLayout(right_file_content)
        right_file_layout.setContentsMargins(5, 5, 5, 5)
        
        self.right_file_list = ImageFileList("后时序图像")
        right_file_layout.addWidget(self.right_file_list)
        self.right_file_list.image_selected.connect(self.load_after_image)
        
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_file_dock)
        
        # 右侧图像视图
        self.right_view = BuildingAnnotationView()
        self.right_view.point_clicked.connect(self.on_point_clicked_after)
        self.right_view.building_selected.connect(lambda id: self.on_building_selected(id, "after"))
        self.right_view.polygon_modified.connect(lambda id, points: self.on_polygon_modified(id, points, "after"))
        right_layout.addWidget(self.right_view)
        
        # 右侧状态标签
        self.right_status = QLabel("后时序: 未加载图像")
        self.right_status.setStyleSheet("font-size: 11px; padding: 5px; border-top: 1px solid #ccc; background-color: #f8f8f8;")
        right_layout.addWidget(self.right_status)
        
        right_container = QWidget()
        right_container.setLayout(right_layout)
        main_splitter.addWidget(right_container)
        
        # 设置左右区域初始比例
        main_splitter.setSizes([900, 900])
        
        # 将主分割器添加到主布局
        main_layout.addWidget(main_splitter, 1)
        
        # 创建顶部工具栏
        self.create_toolbar()
        
        # 创建左侧工具栏（工具模式）
        self.create_left_toolbar()
        
        # 创建右侧工具栏（参数与导出）
        self.create_right_toolbar()
        
        # 创建底部状态栏
        self.statusBar().showMessage("就绪 - 请先加载图像")
        
        # 记录当前字体大小
        self.current_font_size = 14
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 初始化主题状态
        self.dark_mode = False
        
        # 设置样式
        self.update_theme()
    

    
    def resizeEvent(self, event):
        """窗口大小改变时的事件处理"""
        super().resizeEvent(event)
    
    def show_easter_egg(self):
        """显示彩蛋对话框"""
        dialog = EasterEggDialog(self)
        dialog.exec_()
    
    def update_theme(self):
        """更新主题样式"""
        if self.dark_mode:
            # 黑暗模式样式
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e1e;
                }
                QWidget {
                    background-color: #1e1e1e;
                    color: #e0e0e0;
                }
                QDockWidget {
                    font-weight: bold;
                    border: 1px solid #444444;
                    border-radius: 4px;
                    margin: 2px;
                }
                QDockWidget::title {
                    background-color: #2d2d2d;
                    padding: 5px;
                    border-bottom: 1px solid #444444;
                }
                QPushButton {
                    background-color: #333333;
                    border: 1px solid #444444;
                    padding: 12px 16px;
                    border-radius: 4px;
                    font-size: 16px;
                    min-height: 45px;
                    color: #e0e0e0;
                }
                QPushButton:hover {
                    background-color: #444444;
                    border: 1px solid #555555;
                }
                QPushButton:pressed {
                    background-color: #2d2d2d;
                }
                QPushButton:disabled {
                    color: #777777;
                    background-color: #2d2d2d;
                }
                QToolButton {
                    background-color: #333333;
                    border: 1px solid #444444;
                    padding: 12px 16px;
                    border-radius: 4px;
                    font-size: 14px;
                    text-align: left;
                    min-height: 40px;
                    color: #e0e0e0;
                }
                QToolButton:hover {
                    background-color: #444444;
                    border: 1px solid #555555;
                }
                QToolButton:pressed {
                    background-color: #2d2d2d;
                }
                QToolButton:checked {
                    background-color: #2d2d2d;
                    border: 2px solid #ffffff;
                }
                QLabel {
                    color: #e0e0e0;
                    font-size: 14px;
                    font-weight: 500;
                }
                QSlider::groove:horizontal {
                    height: 6px;
                    background-color: #333333;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background-color: #ffffff;
                    width: 16px;
                    height: 16px;
                    margin: -5px 0;
                    border-radius: 8px;
                }
                QRadioButton {
                    padding: 4px;
                    font-size: 12px;
                    color: #e0e0e0;
                }
                QRadioButton::indicator {
                    background-color: #333333;
                    border: 1px solid #444444;
                }
                QRadioButton::indicator:checked {
                    background-color: #ffffff;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #444444;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 10px;
                    font-size: 12px;
                    color: #e0e0e0;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QListWidget {
                    border: 1px solid #444444;
                    background-color: #2d2d2d;
                    font-size: 13px;
                }
                QListWidget::item {
                    padding: 6px;
                    color: #e0e0e0;
                    background-color: #2d2d2d;
                }
                QListWidget::item:selected {
                    background-color: #3a3a3a;
                }
                QListWidget::item:hover {
                    background-color: #444444;
                }
                QLineEdit {
                    border: 1px solid #444444;
                    padding: 5px;
                    border-radius: 3px;
                    font-size: 11px;
                    background-color: #333333;
                    color: #e0e0e0;
                }
                QScrollArea {
                    background-color: #1e1e1e;
                    border: none;
                }
                QTabWidget::pane {
                    background-color: #1e1e1e;
                    border: 1px solid #444444;
                }
                QTabWidget::tab-bar {
                    background-color: #1e1e1e;
                }
                QTabBar::tab {
                    background-color: #333333;
                    color: #e0e0e0;
                    padding: 8px 16px;
                    border: 1px solid #444444;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #1e1e1e;
                    border-bottom-color: #1e1e1e;
                }
                QTabBar::tab:hover {
                    background-color: #444444;
                }
            """)
        else:
            # 光明模式样式
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f0f0f0;
                }
                QWidget {
                    background-color: #f0f0f0;
                    color: #333333;
                }
                QDockWidget {
                    font-weight: bold;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    margin: 2px;
                }
                QDockWidget::title {
                    background-color: #e0e0e0;
                    padding: 5px;
                    border-bottom: 1px solid #cccccc;
                }
                QPushButton {
                    background-color: #ffffff;
                    border: 1px solid #cccccc;
                    padding: 12px 16px;
                    border-radius: 4px;
                    font-size: 16px;
                    min-height: 45px;
                    color: #333333;
                }
                QPushButton:hover {
                    background-color: #f0f0f0;
                    border: 1px solid #999999;
                }
                QPushButton:pressed {
                    background-color: #e0e0e0;
                }
                QPushButton:disabled {
                    color: #999999;
                    background-color: #f5f5f5;
                }
                QToolButton {
                    background-color: #ffffff;
                    border: 1px solid #cccccc;
                    padding: 12px 16px;
                    border-radius: 4px;
                    font-size: 14px;
                    text-align: left;
                    min-height: 40px;
                    color: #333333;
                }
                QToolButton:hover {
                    background-color: #f0f0f0;
                    border: 1px solid #999999;
                }
                QToolButton:pressed {
                    background-color: #e0e0e0;
                }
                QToolButton:checked {
                    background-color: #d0d0d0;
                    border: 2px solid #333333;
                }
                QLabel {
                    color: #333333;
                    font-size: 14px;
                    font-weight: 500;
                }
                QSlider::groove:horizontal {
                    height: 6px;
                    background-color: #e0e0e0;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background-color: #333333;
                    width: 16px;
                    height: 16px;
                    margin: -5px 0;
                    border-radius: 8px;
                }
                QRadioButton {
                    padding: 4px;
                    font-size: 12px;
                    color: #333333;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 10px;
                    font-size: 12px;
                    color: #333333;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QListWidget {
                    border: 1px solid #cccccc;
                    background-color: white;
                    font-size: 13px;
                    color: #333333;
                }
                QListWidget::item {
                    padding: 6px;
                    background-color: white;
                }
                QListWidget::item:selected {
                    background-color: #e0e0e0;
                }
                QListWidget::item:hover {
                    background-color: #f0f0f0;
                }
                QLineEdit {
                    border: 1px solid #cccccc;
                    padding: 5px;
                    border-radius: 3px;
                    font-size: 11px;
                    background-color: white;
                    color: #333333;
                }
            """)
        
        # 更新文件列表的深色模式
        if hasattr(self, 'left_file_list') and self.left_file_list is not None:
            self.left_file_list.set_dark_mode(self.dark_mode)
        if hasattr(self, 'right_file_list') and self.right_file_list is not None:
            self.right_file_list.set_dark_mode(self.dark_mode)
    
    def create_toolbar(self):
        """创建顶部工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        
        # 模型加载
        self.load_model_btn = QToolButton()
        self.load_model_btn.setText("📂 加载SAM模型")
        self.load_model_btn.setToolTip("加载SAM模型文件")
        self.load_model_btn.clicked.connect(self.load_sam_model)
        self.load_model_btn.setStyleSheet("font-size: 14px;")
        toolbar.addWidget(self.load_model_btn)
        
        toolbar.addSeparator()
        
        # 模型状态
        self.model_status = QLabel("❌ 未加载SAM模型")
        self.model_status.setStyleSheet("color: #999999; font-weight: bold; padding: 4px; font-size: 14px;")
        toolbar.addWidget(self.model_status)
        
        # 自由标注
        self.free_annotation_btn = QToolButton()
        self.free_annotation_btn.setText("🖍️ 自由标注")
        self.free_annotation_btn.setToolTip("打开自由标注模式")
        self.free_annotation_btn.clicked.connect(self.open_free_annotation)
        self.free_annotation_btn.setStyleSheet("font-size: 14px;")
        toolbar.addWidget(self.free_annotation_btn)
        
        toolbar.addSeparator()
        
        # 缓存管理
        self.clear_cache_btn = QToolButton()
        self.clear_cache_btn.setText("🗑️ 清空缓存")
        self.clear_cache_btn.setToolTip("清空图像缓存")
        self.clear_cache_btn.clicked.connect(self.clear_image_cache)
        self.clear_cache_btn.setStyleSheet("font-size: 14px;")
        toolbar.addWidget(self.clear_cache_btn)
        
        toolbar.addSeparator()
        
        # 主题切换按钮
        self.theme_btn = QToolButton()
        self.theme_btn.setText("🌙 黑暗模式")
        self.theme_btn.setToolTip("切换深色/浅色主题")
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.theme_btn.setStyleSheet("font-size: 14px;")
        toolbar.addWidget(self.theme_btn)
        
        toolbar.addSeparator()
        
        # 对比模式切换
        self.compare_mode = False
        self.after_image_visible = True
        self.compare_mode_btn = QToolButton()
        self.compare_mode_btn.setText("🔄 左右视图")
        self.compare_mode_btn.setToolTip("切换到叠加对比模式")
        self.compare_mode_btn.clicked.connect(self.toggle_compare_mode)
        self.compare_mode_btn.setStyleSheet("font-size: 14px;")
        toolbar.addWidget(self.compare_mode_btn)
        
        toolbar.addSeparator()
        
        # 后时序图显示/隐藏按钮
        self.toggle_after_btn = QToolButton()
        self.toggle_after_btn.setText("👁️ 显示后时序")
        self.toggle_after_btn.setToolTip("隐藏/显示后时序图像")
        self.toggle_after_btn.clicked.connect(self.toggle_after_visibility)
        self.toggle_after_btn.setStyleSheet("font-size: 14px;")
        self.toggle_after_btn.setEnabled(False)  # 初始禁用，只有在对比模式下可用
        toolbar.addWidget(self.toggle_after_btn)
        
        toolbar.addSeparator()
        
        # 帮助按钮
        self.help_btn = QToolButton()
        self.help_btn.setText("❓ 帮助")
        self.help_btn.setToolTip("显示帮助信息 (F1)")
        self.help_btn.clicked.connect(self.show_help)
        self.help_btn.setStyleSheet("font-size: 14px;")
        toolbar.addWidget(self.help_btn)
        

    
    def toggle_theme(self):
        """切换主题"""
        self.dark_mode = not self.dark_mode
        self.update_theme()
        
        # 更新主题按钮文本
        if self.dark_mode:
            self.theme_btn.setText("☀️ 浅色模式")
        else:
            self.theme_btn.setText("🌙 黑暗模式")
            
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 视图菜单
        view_menu = menubar.addMenu('视图')
        
        # 字体大小子菜单
        font_size_menu = view_menu.addMenu('字体大小')
        
        # 字体大小选项
        font_sizes = [12, 14, 16, 18, 20, 24]
        for size in font_sizes:
            action = QAction(f'{size}px', self)
            action.setCheckable(True)
            if size == self.current_font_size:
                action.setChecked(True)
            action.triggered.connect(lambda checked, s=size: self.change_font_size(s))
            font_size_menu.addAction(action)
        
        # 字体大小调节动作
        decrease_font_action = QAction('减小字体', self)
        decrease_font_action.setShortcut('Ctrl+-')
        decrease_font_action.triggered.connect(self.decrease_font_size)
        view_menu.addAction(decrease_font_action)
        
        increase_font_action = QAction('增大字体', self)
        increase_font_action.setShortcut('Ctrl++')
        increase_font_action.triggered.connect(self.increase_font_size)
        view_menu.addAction(increase_font_action)
        
        # 重置字体大小
        reset_font_action = QAction('重置字体大小', self)
        reset_font_action.setShortcut('Ctrl+0')
        reset_font_action.triggered.connect(lambda: self.change_font_size(14))
        view_menu.addAction(reset_font_action)
        
    def change_font_size(self, size):
        """改变字体大小"""
        self.current_font_size = size
        font = QFont("Microsoft YaHei UI", size)
        app = QApplication.instance()
        app.setFont(font)
        
        # 更新字体大小子菜单的选中状态
        menubar = self.menuBar()
        view_menu = menubar.actions()[0].menu()  # 假设视图菜单是第一个
        font_size_menu = view_menu.actions()[0].menu()  # 假设字体大小子菜单是第一个
        
        for action in font_size_menu.actions():
            if int(action.text().replace('px', '')) == size:
                action.setChecked(True)
            else:
                action.setChecked(False)
    
    def decrease_font_size(self):
        """减小字体大小"""
        if self.current_font_size > 12:
            self.change_font_size(self.current_font_size - 2)
    
    def increase_font_size(self):
        """增大字体大小"""
        if self.current_font_size < 24:
            self.change_font_size(self.current_font_size + 2)
    
    def toggle_compare_mode(self):
        """切换对比模式"""
        self.compare_mode = not self.compare_mode
        
        if self.compare_mode:
            # 切换到叠加对比模式
            self.compare_mode_btn.setText("🔄 叠加对比")
            self.compare_mode_btn.setToolTip("切换回左右视图模式")
            self.toggle_after_btn.setEnabled(True)
            
            # 将后时序图像添加到左视图
            if self.current_image_after is not None and self.right_view.image_item is not None:
                # 创建后时序图像的副本
                self.after_image_item = QGraphicsPixmapItem(self.right_view.image_pixmap)
                self.after_image_item.setZValue(10)  # 放在上层
                self.left_view.scene.addItem(self.after_image_item)
                
                # 隐藏右视图
                self.right_view.hide()
        else:
            # 切换回左右视图模式
            self.compare_mode_btn.setText("🔄 左右视图")
            self.compare_mode_btn.setToolTip("切换到叠加对比模式")
            self.toggle_after_btn.setEnabled(False)
            
            # 移除后时序图像 - 确保对象仍然有效
            if hasattr(self, 'after_image_item'):
                try:
                    # 检查对象是否仍然有效
                    if self.after_image_item is not None:
                        # 在删除前先检查是否在场景中
                        if self.after_image_item.scene() is not None:
                            self.left_view.scene.removeItem(self.after_image_item)
                        del self.after_image_item
                except RuntimeError:
                    # 忽略对象已被删除的错误
                    del self.after_image_item
            
            # 显示右视图
            self.right_view.show()
    
    def toggle_after_visibility(self):
        """切换后时序图的显示/隐藏"""
        if not self.compare_mode:
            return
        
        self.after_image_visible = not self.after_image_visible
        
        if self.after_image_visible:
            self.toggle_after_btn.setText("👁️ 显示后时序")
            self.toggle_after_btn.setToolTip("隐藏后时序图像")
            try:
                if hasattr(self, 'after_image_item') and self.after_image_item is not None:
                    # 检查对象是否仍然有效
                    self.after_image_item.show()
            except RuntimeError:
                # 对象已被删除，重置状态
                if hasattr(self, 'after_image_item'):
                    del self.after_image_item
        else:
            self.toggle_after_btn.setText("👁️ 隐藏后时序")
            self.toggle_after_btn.setToolTip("显示后时序图像")
            try:
                if hasattr(self, 'after_image_item') and self.after_image_item is not None:
                    # 检查对象是否仍然有效
                    self.after_image_item.hide()
            except RuntimeError:
                # 对象已被删除，重置状态
                if hasattr(self, 'after_image_item'):
                    del self.after_image_item
    
    def delete_selected_building(self):
        """删除当前选中的建筑物"""
        if not hasattr(self, 'current_selected_building') or not self.current_selected_building:
            return
        
        try:
            # 获取建筑物信息并验证
            if not all(key in self.current_selected_building for key in ["id", "type", "image_type"]):
                return
            
            building_id = self.current_selected_building["id"]
            building_type = self.current_selected_building["type"]
            image_type = self.current_selected_building["image_type"]
            
            # 获取对应的视图
            view = self.left_view if image_type == "before" else self.right_view
            if not hasattr(view, 'building_items'):
                return
            
            if building_type == "sam":
                # 删除SAM分割的建筑物
                sam = getattr(self, f'sam_{image_type}', None)
                if sam and hasattr(sam, 'buildings'):
                    # 找到并删除对应的建筑物
                    original_count = len(sam.buildings)
                    sam.buildings = [b for b in sam.buildings if b["id"] != building_id]
                    if len(sam.buildings) < original_count:
                        # 只有在确实删除了建筑物时才重新生成图形项
                        view.generate_building_items(sam.buildings)
            elif building_type == "manual":
                # 删除手动绘制的建筑物
                view.delete_building(building_id)
            
            # 视图会自动更新，不需要额外调用update_buildings_display方法
            
            # 清除选中状态
            self.current_selected_building = None
            
            # 更新按钮状态
            if hasattr(self, 'delete_btn'):
                self.delete_btn.setEnabled(False)
            
            # 更新建筑物信息面板
            building_type_name = "先时序" if image_type == "before" else "后时序"
            self.update_building_info(None, image_type, building_type_name)
            
            # 更新统计信息
            self.update_stats()
        except Exception as e:
            # 捕获所有异常，防止程序闪退
            print(f"删除建筑物时发生错误: {e}")
            # 确保状态正确重置
            self.current_selected_building = None
            if hasattr(self, 'delete_btn'):
                self.delete_btn.setEnabled(False)
    
    def create_left_toolbar(self):
        """创建左侧工具栏（工具模式）"""
        left_dock = ToolBarDockWidget("工具模式")
        left_content = left_dock.widget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)
        
        # 工具模式选择（标题字体稍大一点）
        tool_mode_label = QLabel("选择工具模式:")
        tool_mode_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        left_layout.addWidget(tool_mode_label)
        
        self.tool_group = QButtonGroup(self)
        
        self.sam_mode_btn = QToolButton()
        self.sam_mode_btn.setText("🔍 SAM分割")
        self.sam_mode_btn.setCheckable(True)
        self.sam_mode_btn.setChecked(False)
        self.sam_mode_btn.clicked.connect(lambda: self.set_tool_mode("sam"))
        self.sam_mode_btn.setToolTip("SAM分割模式：点击建筑物自动分割")
        left_layout.addWidget(self.sam_mode_btn)
        
        self.brush_mode_btn = QToolButton()
        self.brush_mode_btn.setText("✏️ 画笔绘制")
        self.brush_mode_btn.setCheckable(True)
        self.brush_mode_btn.setChecked(True)  # 默认选中画笔模式
        self.brush_mode_btn.clicked.connect(lambda: self.set_tool_mode("brush"))
        self.brush_mode_btn.setToolTip("画笔绘制模式：手动绘制多边形")
        left_layout.addWidget(self.brush_mode_btn)
        
        self.edit_mode_btn = QToolButton()
        self.edit_mode_btn.setText("📝 编辑模式")
        self.edit_mode_btn.setCheckable(True)
        self.edit_mode_btn.clicked.connect(lambda: self.set_tool_mode("edit"))
        self.edit_mode_btn.setToolTip("编辑模式：选择建筑物，拖动顶点调整")
        left_layout.addWidget(self.edit_mode_btn)
        
        # 将按钮添加到按钮组
        self.tool_group.addButton(self.sam_mode_btn)
        self.tool_group.addButton(self.brush_mode_btn)
        self.tool_group.addButton(self.edit_mode_btn)
        
        # 初始化画笔模式（因为默认选中了画笔按钮）
        self.set_tool_mode("brush")
        
        left_layout.addSpacing(10)
        
        # 操作按钮
        mark_label = QLabel("标注操作:")
        mark_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        left_layout.addWidget(mark_label)
        
        self.undo_btn = QToolButton()
        self.undo_btn.setText("↶ 撤销 (Ctrl+Z)")
        self.undo_btn.setToolTip("撤销最近的操作")
        self.undo_btn.clicked.connect(self.undo_last_action)
        self.undo_btn.setEnabled(False)
        left_layout.addWidget(self.undo_btn)
        
        self.clear_btn = QToolButton()
        self.clear_btn.setText("🗑️ 清除所有")
        self.clear_btn.setToolTip("清除所有标注")
        self.clear_btn.clicked.connect(self.clear_all_annotations)
        left_layout.addWidget(self.clear_btn)
        
        self.delete_btn = QToolButton()
        self.delete_btn.setText("🗑️ 删除选中")
        self.delete_btn.setToolTip("删除当前选中的建筑物")
        self.delete_btn.clicked.connect(self.delete_selected_building)
        self.delete_btn.setEnabled(False)
        left_layout.addWidget(self.delete_btn)
        
        left_layout.addStretch()
        
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)
    
    def create_right_toolbar(self):
        """创建右侧工具栏（参数与导出）"""
        right_dock = ToolBarDockWidget("参数与导出")
        right_content = right_dock.widget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)
        
        # 参数设置（右侧标题字体稍大）
        params_label = QLabel("参数设置:")
        params_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        right_layout.addWidget(params_label)
        
        # 置信度阈值
        conf_label = QLabel("置信度阈值:")
        conf_label.setStyleSheet("font-size: 14px;")
        right_layout.addWidget(conf_label)
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(30, 95)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        right_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("0.50")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.confidence_label)
        
        # 多边形精细度
        precision_title = QLabel("多边形精细度:")
        precision_title.setStyleSheet("font-size: 14px;")
        right_layout.addWidget(precision_title)
        
        self.precision_slider = QSlider(Qt.Horizontal)
        self.precision_slider.setRange(1, 10)
        self.precision_slider.setValue(self.precision_level)
        self.precision_slider.valueChanged.connect(self.update_precision)
        right_layout.addWidget(self.precision_slider)
        
        self.precision_label = QLabel(f"{self.precision_level} (中等)")
        self.precision_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.precision_label)
        
        # 精细度说明
        precision_info_layout = QHBoxLayout()
        self.low_precision_label = QLabel("粗略")
        self.low_precision_label.setAlignment(Qt.AlignLeft)
        self.low_precision_label.setStyleSheet("color: #666; font-size: 10px;")
        
        self.high_precision_label = QLabel("精细")
        self.high_precision_label.setAlignment(Qt.AlignRight)
        self.high_precision_label.setStyleSheet("color: #666; font-size: 10px;")
        
        precision_info_layout.addWidget(self.low_precision_label)
        precision_info_layout.addStretch()
        precision_info_layout.addWidget(self.high_precision_label)
        
        right_layout.addLayout(precision_info_layout)
        
        right_layout.addSpacing(10)
        
        # 导出按钮
        export_label = QLabel("导出操作:")
        export_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        right_layout.addWidget(export_label)
        
        self.export_btn = QToolButton()
        self.export_btn.setText("💾 导出二值图")
        self.export_btn.setToolTip("导出先时序图像的二值图")
        self.export_btn.clicked.connect(self.export_binary_mask)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("font-size: 14px;")
        right_layout.addWidget(self.export_btn)
        
        right_layout.addStretch()
        
        # 统计信息
        self.stats_label = QLabel("先时序标注: 0 | 后时序标注: 0")
        self.stats_label.setStyleSheet("font-weight: bold; margin-top: 10px; padding: 5px; background-color: #e8e8e8; border-radius: 3px;")
        right_layout.addWidget(self.stats_label)
        
        # 建筑物信息
        info_group = QGroupBox("建筑物信息")
        info_layout = QVBoxLayout(info_group)
        
        info_grid = QGridLayout()
        info_grid.addWidget(QLabel("ID:"), 0, 0)
        self.building_id_label = QLabel("-")
        info_grid.addWidget(self.building_id_label, 0, 1)
        
        info_grid.addWidget(QLabel("面积:"), 1, 0)
        self.building_area_label = QLabel("- px²")
        info_grid.addWidget(self.building_area_label, 1, 1)
        
        info_grid.addWidget(QLabel("置信度:"), 2, 0)
        self.building_score_label = QLabel("-")
        info_grid.addWidget(self.building_score_label, 2, 1)
        
        info_grid.addWidget(QLabel("顶点数:"), 3, 0)
        self.vertices_label = QLabel("-")
        info_grid.addWidget(self.vertices_label, 3, 1)
        
        info_grid.addWidget(QLabel("类型:"), 4, 0)
        self.building_type_label = QLabel("-")
        info_grid.addWidget(self.building_type_label, 4, 1)
        
        info_layout.addLayout(info_grid)
        right_layout.addWidget(info_group)
        
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)
    
    def show_loading_dialog(self, title="加载中..."):
        """显示加载对话框"""
        if self.loading_dialog is None:
            self.loading_dialog = QDialog(self)
            self.loading_dialog.setWindowTitle(title)
            self.loading_dialog.setFixedSize(300, 100)
            
            layout = QVBoxLayout(self.loading_dialog)
            self.loading_label = QLabel("正在加载图像...")
            layout.addWidget(self.loading_label)
            
            self.loading_progress = QProgressBar()
            self.loading_progress.setRange(0, 100)
            layout.addWidget(self.loading_progress)
        
        self.loading_dialog.show()
        QApplication.processEvents()
    
    def hide_loading_dialog(self):
        """隐藏加载对话框"""
        if self.loading_dialog:
            self.loading_dialog.hide()
    
    def update_loading_progress(self, message, progress):
        """更新加载进度"""
        if self.loading_dialog:
            self.loading_label.setText(message)
            self.loading_progress.setValue(progress)
            QApplication.processEvents()
    
    def show_help(self):
        """显示帮助信息"""
        # 创建自定义帮助对话框
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("帮助")
        help_dialog.setFixedSize(700, 600)
        
        # 设置窗口样式（只设置文本编辑框样式，不改变按钮颜色）
        help_dialog.setStyleSheet("""
        QTextEdit {
            background-color: white;
            border: 1px solid #cccccc;
            font-family: Arial, sans-serif;
            font-size: 16px;
            padding: 10px;
        }
        """)
        
        # 创建垂直布局
        layout = QVBoxLayout(help_dialog)
        
        # 创建文本编辑框显示帮助内容
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml(
            "<b>SAM建筑物像素级标注工具 - 遥感变化检测版</b><br><br>"
            "<b>工作流程:</b><br>"
            "1. 点击'加载SAM模型'选择.pth模型文件<br>"
            "2. 分别在左右两侧的'选择文件夹'按钮导入图像文件夹<br>"
            "   或者使用'+'按钮添加单张图片<br>"
            "3. 在文件列表中双击图像加载到对应时序区域<br>"
            "4. 选择工具模式进行标注：<br>"
            "   • SAM分割模式：点击建筑物自动分割<br>"
            "   • 画笔绘制模式：手动绘制多边形<br>"
            "   • 编辑模式：选择建筑物，拖动顶点调整<br>"
            "5. 在先时序图像上进行标注<br>"
            "6. 点击'导出二值图'保存标签<br><br>"
            "<b>新界面特性:</b><br>"
            "• 侧边栏设计，最大化绘图区域<br>"
            "• 工具栏可停靠、可拖动、可折叠<br>"
            "• 1像素边框和顶点，减少视觉干扰<br>"
            "• 鼠标中键拖动图像<br>"
            "• 编辑模式下右键取消选中<br><br>"
            "<b>快捷键:</b><br>"
            "• 滚轮：缩放图像<br>"
            "• 鼠标中键：拖动图像<br>"
            "• +/-：缩放图像<br>"
            "• F：适应窗口<br>"
            "• Delete：删除选中的建筑物<br>"
            "• Ctrl+Z：撤销最近的操作<br>"
            "• F1：显示帮助信息<br>"
        )
        
        # 创建底部水平布局
        bottom_layout = QHBoxLayout()
        
        # 添加彩蛋按钮到左下角
        self.easter_egg_button = QPushButton("🥚 彩蛋", help_dialog)
        self.easter_egg_button.setToolTip("发现隐藏的彩蛋")
        self.easter_egg_button.clicked.connect(self.show_easter_egg)
        self.easter_egg_button.setFixedSize(100, 35)  # 增大按钮大小
        self.easter_egg_button.setStyleSheet("""QPushButton {
            font-family: Arial, sans-serif;
            font-size: 14px;
            color: #333333;
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #f0f0f0;
        }
        QPushButton:pressed {
            background-color: #e0e0e0;
        }""")
        
        # 添加弹簧将按钮推到左侧
        bottom_layout.addWidget(self.easter_egg_button)
        bottom_layout.addStretch()
        
        # 添加关闭按钮
        close_button = QPushButton("关闭", help_dialog)
        close_button.clicked.connect(help_dialog.accept)
        bottom_layout.addWidget(close_button)
        
        # 添加控件到布局
        layout.addWidget(help_text)
        layout.addLayout(bottom_layout)
        
        # 设置间距和边距
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 显示对话框
        help_dialog.exec_()
    
    def open_free_annotation(self):
        """打开自由标注模式"""
        # 检查是否已加载SAM模型
        if not hasattr(self, 'sam') or not self.sam:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先加载SAM模型")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
            msg_box.exec_()
            return
        
        # 确定打开哪个图像
        images = []
        if hasattr(self, 'sam_before') and self.sam_before and self.sam_before.image_path:
            images.append(("先时序", self.sam_before.image_path, "before"))
        if hasattr(self, 'sam_after') and self.sam_after and self.sam_after.image_path:
            images.append(("后时序", self.sam_after.image_path, "after"))
        
        if not images:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("警告")
            msg_box.setText("请先加载图像")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
            msg_box.exec_()
            return
        
        # 如果有多个图像，让用户选择
        if len(images) > 1:
            image_names = [f"{name}: {Path(path).name}" for name, path, _ in images]
            selected, ok = QInputDialog.getItem(self, "选择图像", "请选择要标注的图像:", image_names, 0, False)
            if not ok:
                return
            
            selected_index = image_names.index(selected)
            selected_name, selected_path, selected_type = images[selected_index]
        else:
            selected_name, selected_path, selected_type = images[0]
        
        # 创建自由标注对话框
        free_dialog = FreeAnnotationDialog(self)
        free_dialog.set_image(selected_path, self.sam.predictor, selected_type)
        
        # 连接同步信号
        free_dialog.building_added.connect(self.sync_building_from_free)
        free_dialog.building_deleted.connect(self.sync_delete_from_free)
        free_dialog.polygon_updated.connect(self.sync_polygon_from_free)
        
        # 存储引用
        self.free_annotation_dialogs.append(free_dialog)
        
        # 当对话框关闭时移除引用
        free_dialog.finished.connect(lambda: self.free_annotation_dialogs.remove(free_dialog))
        
        # 同步当前标注到自由标注窗口
        self.sync_to_free_annotation(free_dialog, selected_type)
        
        free_dialog.exec_()
    
    def sync_to_free_annotation(self, free_dialog, image_type):
        """同步当前标注到自由标注窗口"""
        # 根据图像类型获取对应的SAM实例和视图
        if image_type == "before":
            sam_instance = self.sam_before if hasattr(self, 'sam_before') else None
            view = self.left_view
        else:
            sam_instance = self.sam_after if hasattr(self, 'sam_after') else None
            view = self.right_view
        
        if not sam_instance:
            return
        
        # 同步SAM建筑物
        for building in sam_instance.buildings:
            free_dialog.sam_instance.buildings.append(copy.deepcopy(building))
            free_dialog.view.add_building(building)
        
        # 同步手动建筑物
        for building_id, item in view.building_items.items():
            if building_id < 0:  # 手动绘制的建筑物
                free_dialog.view.add_manual_polygon(item.get_points())
    
    def sync_building_from_free(self, building, image_type):
        """从自由标注窗口同步建筑物"""
        # 保存状态
        self.save_state(image_type)
        
        # 根据图像类型获取对应的SAM实例和视图
        if image_type == "before":
            sam_instance = self.sam_before if hasattr(self, 'sam_before') else None
            view = self.left_view
        else:
            sam_instance = self.sam_after if hasattr(self, 'sam_after') else None
            view = self.right_view
        
        if not sam_instance:
            return
        
        # 如果是SAM分割的建筑物（ID >= 0）
        if building["id"] >= 0:
            # 检查是否已存在
            existing = False
            for b in sam_instance.buildings:
                if b["id"] == building["id"]:
                    existing = True
                    break
            
            if not existing:
                # 添加到SAM实例
                sam_instance.buildings.append(copy.deepcopy(building))
                # 更新ID计数器
                sam_instance.current_building_id = max(sam_instance.current_building_id, building["id"] + 1)
                # 添加到视图
                view.add_building(building)
        else:
            # 手动绘制的建筑物
            view.add_manual_polygon(building["points"])
        
        # 更新统计信息
        self.update_stats()
    
    def sync_delete_from_free(self, building_id, image_type):
        """从自由标注窗口同步删除"""
        # 保存状态
        self.save_state(image_type)
        
        # 根据图像类型获取对应的SAM实例和视图
        if image_type == "before":
            sam_instance = self.sam_before if hasattr(self, 'sam_before') else None
            view = self.left_view
        else:
            sam_instance = self.sam_after if hasattr(self, 'sam_after') else None
            view = self.right_view
        
        if not sam_instance:
            return
        
        # 特殊ID：清除所有
        if building_id == -999:
            sam_instance.clear_all_buildings()
            view.clear_all_buildings()
        else:
            # 删除指定建筑物
            if building_id >= 0:  # SAM分割的建筑物
                sam_instance.delete_building(building_id)
            view.delete_building(building_id)
        
        # 更新统计信息
        self.update_stats()
    
    def sync_polygon_from_free(self, building_id, points, image_type):
        """从自由标注窗口同步多边形修改"""
        # 保存状态
        self.save_state(image_type)
        
        # 根据图像类型获取对应的视图
        if image_type == "before":
            view = self.left_view
        else:
            view = self.right_view
        
        # 查找并更新对应的多边形
        for item_building_id, item in view.building_items.items():
            if item_building_id == building_id:
                # 创建点列表
                point_objects = [QPointF(x, y) for x, y in points]
                
                # 更新顶点
                for i in range(min(len(item.points), len(point_objects))):
                    item.points[i] = point_objects[i]
                
                # 更新多边形形状
                item.update_polygon()
                
                # 更新对应顶点项的位置
                if i < len(item.vertex_items):
                    item.vertex_items[i].setPos(point_objects[i])
                
                break
    
    def clear_image_cache(self):
        """清空图像缓存"""
        ImageCache.clear()
        self.statusBar().showMessage("已清空图像缓存")
    
    def load_sam_model(self):
        """加载SAM模型"""
        # 打开文件对话框选择模型文件
        model_file, _ = QFileDialog.getOpenFileName(
            self, "选择SAM模型文件", "", "模型文件 (*.pth)"
        )
        
        if not model_file:
            return
        
        try:
            # 显示加载状态
            self.statusBar().showMessage("正在加载SAM模型...")
            QApplication.processEvents()
            
            # 初始化SAM模型
            self.sam = PixelPrecisionSAM()
            
            # 手动加载模型
            print("正在加载模型文件...")
            model_type = "vit_b"  # 默认模型类型
            self.sam.sam = sam_model_registry[model_type](checkpoint=model_file)
            self.sam.sam.to(device=self.sam.device)
            self.sam.predictor = SamPredictor(self.sam.sam)
            print("SAM模型加载成功!")
            
            # 更新状态
            self.model_status.setText("✅ SAM模型已加载")
            self.model_status.setStyleSheet("color: #00aa00; font-weight: bold; padding: 4px; font-size: 14px;")
            
            # 启用SAM模式按钮
            self.sam_mode_btn.setEnabled(True)
            
            self.statusBar().showMessage("SAM模型加载成功")
        except Exception as e:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("错误")
            msg_box.setText(f"加载模型失败:\n{str(e)}")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
            msg_box.exec_()
            
            # 更新状态
            self.model_status.setText("❌ 加载模型失败")
            self.model_status.setStyleSheet("color: #ff0000; font-weight: bold; padding: 4px; font-size: 14px;")
            
            self.statusBar().showMessage("加载模型失败")
    
    def save_state(self, image_type):
        """保存当前状态到撤销栈"""
        import time
        
        if image_type == "before":
            if not hasattr(self, 'sam_before'):
                return
            
            # 获取当前SAM建筑物数据
            sam_buildings = copy.deepcopy(self.sam_before.buildings) if self.sam_before else []
            
            # 获取当前视图中的手动建筑物数据
            manual_buildings = []
            for building_id, item in self.left_view.building_items.items():
                if building_id < 0:  # 手动绘制的建筑物
                    manual_buildings.append({
                        'id': building_id,
                        'points': item.get_points()
                    })
            
            # 创建状态对象
            state = {
                'sam_buildings': sam_buildings,
                'manual_buildings': manual_buildings,
                'current_building_id': self.sam_before.current_building_id if self.sam_before else 0
            }
            
            # 添加到先时序撤销栈
            self.undo_stack_before.append(state)
            
            # 限制栈大小
            if len(self.undo_stack_before) > self.max_undo_steps:
                self.undo_stack_before.pop(0)
                
            # 添加到统一撤销栈（按时间顺序）
            self.undo_stack.append({
                'timestamp': time.time(),
                'image_type': 'before',
                'state_index': len(self.undo_stack_before) - 1
            })
            
        else:  # after
            if not hasattr(self, 'sam_after'):
                return
            
            # 获取当前SAM建筑物数据
            sam_buildings = copy.deepcopy(self.sam_after.buildings) if self.sam_after else []
            
            # 获取当前视图中的手动建筑物数据
            manual_buildings = []
            for building_id, item in self.right_view.building_items.items():
                if building_id < 0:  # 手动绘制的建筑物
                    manual_buildings.append({
                        'id': building_id,
                        'points': item.get_points()
                    })
            
            # 创建状态对象
            state = {
                'sam_buildings': sam_buildings,
                'manual_buildings': manual_buildings,
                'current_building_id': self.sam_after.current_building_id if self.sam_after else 0
            }
            
            # 添加到后时序撤销栈
            self.undo_stack_after.append(state)
            
            # 限制栈大小
            if len(self.undo_stack_after) > self.max_undo_steps:
                self.undo_stack_after.pop(0)
                
            # 添加到统一撤销栈（按时间顺序）
            self.undo_stack.append({
                'timestamp': time.time(),
                'image_type': 'after',
                'state_index': len(self.undo_stack_after) - 1
            })
        
        # 限制统一撤销栈大小
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
            
        # 更新撤销按钮状态
        self.update_undo_button_state()
    
    def undo_last_action(self):
        """撤销最近的操作"""
        # 检查统一撤销栈是否为空
        if not self.undo_stack:
            return
        
        # 获取最近的操作
        latest_operation = self.undo_stack.pop()
        image_type = latest_operation['image_type']
        
        # 根据图像类型撤销操作
        if image_type == "before":
            if len(self.undo_stack_before) > 0:
                self.undo_stack_before.pop()
                # 如果还有状态，恢复到上一个状态，否则清空
                if len(self.undo_stack_before) > 0:
                    self.restore_state_from_index("before", len(self.undo_stack_before) - 1)
                else:
                    self.clear_all_buildings("before")
        else:  # after
            if len(self.undo_stack_after) > 0:
                self.undo_stack_after.pop()
                # 如果还有状态，恢复到上一个状态，否则清空
                if len(self.undo_stack_after) > 0:
                    self.restore_state_from_index("after", len(self.undo_stack_after) - 1)
                else:
                    self.clear_all_buildings("after")
        
        # 更新撤销按钮状态
        self.update_undo_button_state()
        
        # 更新统计信息
        self.update_stats()
        
        self.statusBar().showMessage(f"已撤销{image_type}的最近操作")
    
    def restore_state(self, image_type):
        """从撤销栈恢复状态"""
        self.restore_state_from_index(image_type, -1)
        
    def restore_state_from_index(self, image_type, index):
        """从撤销栈指定索引恢复状态"""
        if image_type == "before":
            if not self.undo_stack_before or index < 0 or index >= len(self.undo_stack_before):
                return
            
            # 获取指定状态
            state = self.undo_stack_before[index]
            
            # 恢复SAM建筑物
            if hasattr(self, 'sam_before') and self.sam_before:
                self.sam_before.buildings = copy.deepcopy(state['sam_buildings'])
                self.sam_before.current_building_id = state['current_building_id']
            
            # 清除视图中的所有建筑物
            self.left_view.clear_all_buildings()
            
            # 恢复SAM建筑物到视图
            for building in state['sam_buildings']:
                self.left_view.add_building(building)
            
            # 恢复手动建筑物到视图
            for building in state['manual_buildings']:
                self.left_view.add_manual_polygon(building['points'])
            
            # 同步到所有打开的自由标注窗口
            self.sync_to_all_free_dialogs("before")
            
        else:  # after
            if not self.undo_stack_after or index < 0 or index >= len(self.undo_stack_after):
                return
            
            # 获取指定状态
            state = self.undo_stack_after[index]
            
            # 恢复SAM建筑物
            if hasattr(self, 'sam_after') and self.sam_after:
                self.sam_after.buildings = copy.deepcopy(state['sam_buildings'])
                self.sam_after.current_building_id = state['current_building_id']
            
            # 清除视图中的所有建筑物
            self.right_view.clear_all_buildings()
            
            # 恢复SAM建筑物到视图
            for building in state['sam_buildings']:
                self.right_view.add_building(building)
            
            # 恢复手动建筑物到视图
            for building in state['manual_buildings']:
                self.right_view.add_manual_polygon(building['points'])
            
            # 同步到所有打开的自由标注窗口
            self.sync_to_all_free_dialogs("after")
        
        # 更新撤销按钮状态
        self.update_undo_button_state()
        
        # 更新统计信息
        self.update_stats()
        
        self.statusBar().showMessage(f"已恢复{image_type}的状态")
        
    def clear_all_buildings(self, image_type):
        """清空指定图像类型的所有建筑物"""
        if image_type == "before":
            # 清除SAM建筑物
            if hasattr(self, 'sam_before') and self.sam_before:
                self.sam_before.buildings = []
                self.sam_before.current_building_id = 0
            
            # 清除视图中的所有建筑物
            self.left_view.clear_all_buildings()
            
            # 同步到所有打开的自由标注窗口
            self.sync_to_all_free_dialogs("before")
        else:  # after
            # 清除SAM建筑物
            if hasattr(self, 'sam_after') and self.sam_after:
                self.sam_after.buildings = []
                self.sam_after.current_building_id = 0
            
            # 清除视图中的所有建筑物
            self.right_view.clear_all_buildings()
            
            # 同步到所有打开的自由标注窗口
            self.sync_to_all_free_dialogs("after")
    
    def sync_to_all_free_dialogs(self, image_type):
        """同步到所有打开的自由标注窗口"""
        for dialog in self.free_annotation_dialogs:
            if dialog.image_type == image_type:
                # 清除自由标注窗口
                if dialog.sam_instance:
                    dialog.sam_instance.clear_all_buildings()
                dialog.view.clear_all_buildings()
                
                # 重新同步
                self.sync_to_free_annotation(dialog, image_type)
    
    def update_undo_button_state(self):
        """更新撤销按钮状态"""
        can_undo = len(self.undo_stack) > 0
        
        self.undo_btn.setEnabled(can_undo)
    
    def update_precision(self, value):
        """更新多边形精细度 (1-10)"""
        self.precision_level = value
        
        # 更新标签显示
        if value <= 3:
            level_text = f"{value} (粗略)"
        elif value <= 5:
            level_text = f"{value} (中等)"
        elif value <= 8:
            level_text = f"{value} (精细)"
        else:
            level_text = f"{value} (极精细)"
        
        self.precision_label.setText(level_text)
        
        # 更新两个SAM实例的精细度
        if hasattr(self, 'sam_before') and self.sam_before:
            self.sam_before.set_precision_level(value)
        
        if hasattr(self, 'sam_after') and self.sam_after:
            self.sam_after.set_precision_level(value)
        
        # 更新所有自由标注对话框中的SAM实例精细度
        for dialog in self.free_annotation_dialogs:
            if hasattr(dialog, 'sam_instance') and dialog.sam_instance:
                dialog.sam_instance.set_precision_level(value)
        
        # 显示状态消息
        precision_desc = {
            1: "极粗略 (顶点最少)",
            2: "非常粗略",
            3: "粗略",
            4: "偏粗略",
            5: "中等",
            6: "偏精细",
            7: "精细",
            8: "非常精细",
            9: "极精细",
            10: "极精细 (顶点最多)"
        }
        
        self.statusBar().showMessage(f"多边形精细度设置为: {value} - {precision_desc.get(value, '中等')}")
    

    
    def load_before_image(self, image_path):
        """加载先时序图像"""
        self.load_image_async(image_path, "before")
    
    def load_after_image(self, image_path):
        """加载后时序图像"""
        self.load_image_async(image_path, "after")
    
    def load_image_async(self, image_path, image_type):
        """异步加载图像"""
        # 显示加载对话框
        self.show_loading_dialog(f"加载{image_type}图像")
        
        # 创建加载线程
        self.load_thread = ImageLoadThread(image_path)
        self.load_thread.image_loaded.connect(lambda path, data: self.on_image_loaded(path, data, image_type))
        self.load_thread.loading_progress.connect(self.update_loading_progress)
        self.load_thread.start()
    
    def on_image_loaded(self, image_path, image_array, image_type):
        """图像加载完成"""
        try:
            # 隐藏加载对话框
            self.hide_loading_dialog()
            
            if image_type == "before":
                # 修复叠加视图下的闪退问题：如果处于叠加模式且存在后时序图像项，在加载新图像前移除它
                if self.compare_mode and hasattr(self, 'after_image_item'):
                    try:
                        # 检查对象是否仍然有效
                        if self.after_image_item is not None:
                            # 在删除前先检查是否在场景中
                            if self.after_image_item.scene() is not None:
                                self.left_view.scene.removeItem(self.after_image_item)
                        del self.after_image_item
                    except RuntimeError:
                        # 忽略对象已被删除的错误
                        if hasattr(self, 'after_image_item'):
                            del self.after_image_item
                
                # 加载到视图
                self.left_view.load_image(image_array)
                self.current_image_before = image_array
                self.left_status.setText(f"先时序: {Path(image_path).name} ({image_array.shape[1]}×{image_array.shape[0]})")
                
                # 保存图像路径
                self.image_path_before = image_path
                
                # 清空撤销栈
                self.undo_stack_before.clear()
                
                # 设置sam_before为None，避免后续代码报错
                self.sam_before = None
                
            else:  # after
                # 加载到视图
                self.right_view.load_image(image_array)
                self.current_image_after = image_array
                self.right_status.setText(f"后时序: {Path(image_path).name} ({image_array.shape[1]}×{image_array.shape[0]})")
                
                # 保存图像路径
                self.image_path_after = image_path
                
                # 清空撤销栈
                self.undo_stack_after.clear()
                
                # 设置sam_after为None，避免后续代码报错
                self.sam_after = None
                
                # 修复叠加视图下的闪退问题：如果处于叠加模式且存在旧的after_image_item，移除它
                if self.compare_mode and hasattr(self, 'after_image_item'):
                    try:
                        # 检查对象是否仍然有效
                        if self.after_image_item is not None:
                            # 在删除前先检查是否在场景中
                            if self.after_image_item.scene() is not None:
                                self.left_view.scene.removeItem(self.after_image_item)
                        del self.after_image_item
                    except RuntimeError:
                        # 忽略对象已被删除的错误
                        if hasattr(self, 'after_image_item'):
                            del self.after_image_item
                    
                # 如果处于叠加模式，重新创建after_image_item
                if self.compare_mode:
                    # 在叠加模式下更新后时序图像
                    self.after_image_item = QGraphicsPixmapItem(QPixmap.fromImage(QImage(image_array.data, image_array.shape[1], image_array.shape[0], image_array.shape[1] * 3, QImage.Format_RGB888)))
                    self.left_view.scene.addItem(self.after_image_item)
            
            # 启用导出按钮
            if self.current_image_before is not None:
                self.export_btn.setEnabled(True)
            
            # 标记当前操作的图片
            if image_type == "before":
                if hasattr(self, 'left_file_list'):
                    self.left_file_list.mark_current_image(image_path)
            else:
                if hasattr(self, 'right_file_list'):
                    self.right_file_list.mark_current_image(image_path)
            
            # 更新撤销按钮状态
            self.update_undo_button_state()
            
            self.statusBar().showMessage(f"已加载{image_type}图像: {Path(image_path).name}")
            
        except Exception as e:
            self.hide_loading_dialog()
            msg_box = QMessageBox()
            msg_box.setWindowTitle("错误")
            msg_box.setText(f"加载图像失败:\n{str(e)}")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
            msg_box.exec_()
    
    def set_tool_mode(self, mode):
        """设置工具模式"""
        if mode == "sam":
            # 检查是否已加载SAM模型
            if not hasattr(self, 'sam') or not self.sam:
                msg_box = QMessageBox()
                msg_box.setWindowTitle("提示")
                msg_box.setText("请先加载SAM模型")
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
                msg_box.exec_()
                mode = "brush"
            else:
                self.left_view.set_brush_mode(False)
                self.left_view.set_mode("segment")
                self.right_view.set_brush_mode(False)
                self.right_view.set_mode("segment")
                self.statusBar().showMessage("SAM分割模式：点击建筑物自动分割")
        elif mode == "brush":
            self.left_view.set_brush_mode(True)
            self.left_view.set_mode("segment")
            self.right_view.set_brush_mode(True)
            self.right_view.set_mode("segment")
            self.statusBar().showMessage("画笔绘制模式：左键添加顶点，右键完成绘制")
        elif mode == "edit":
            self.left_view.set_mode("edit")
            self.left_view.set_brush_mode(False)
            self.right_view.set_mode("edit")
            self.right_view.set_brush_mode(False)
            self.statusBar().showMessage("编辑模式：选择建筑物，拖动顶点进行微调")
    
    def on_point_clicked_before(self, x, y):
        """处理先时序图像点击事件"""
        # 在操作前保存状态
        self.save_state("before")
        self.on_point_clicked(x, y, "before")
        
        # 同步到所有打开的自由标注窗口
        self.sync_main_to_free_dialogs("before")
    
    def on_point_clicked_after(self, x, y):
        """处理后时序图像点击事件"""
        # 在操作前保存状态
        self.save_state("after")
        self.on_point_clicked(x, y, "after")
        
        # 同步到所有打开的自由标注窗口
        self.sync_main_to_free_dialogs("after")
    
    def sync_main_to_free_dialogs(self, image_type):
        """从主窗口同步到自由标注窗口"""
        for dialog in self.free_annotation_dialogs:
            if dialog.image_type == image_type:
                # 清除自由标注窗口
                if dialog.sam_instance:
                    dialog.sam_instance.clear_all_buildings()
                dialog.view.clear_all_buildings()
                
                # 重新同步
                self.sync_to_free_annotation(dialog, image_type)
    
    def on_point_clicked(self, x, y, image_type):
        """处理点击事件"""
        # 检查是否是SAM模式
        if self.sam_mode_btn.isChecked() and hasattr(self, 'sam') and self.sam:
            # 检查当前工具模式是否为SAM
            if image_type == "before":
                # 加载先时序图像到SAM
                if not hasattr(self, 'sam_before') or self.sam_before is None or self.sam_before.image_path != self.image_path_before:
                    # 直接使用现有的SAM预测器，不需要创建新的PixelPrecisionSAM实例
                    self.sam_before = PixelPrecisionSAM()
                    self.sam_before.predictor = self.sam.predictor
                    # 直接设置图像，不调用set_image方法避免重新加载模型
                    self.sam_before.image = self.current_image_before
                    self.sam_before.image_path = self.image_path_before
                    # 重置标注数据
                    self.sam_before.buildings = []
                    self.sam_before.current_building_id = 0
                    # 设置参数
                    self.sam_before.set_precision_level(self.precision_level)
                    self.sam_before.prediction_confidence = self.confidence_slider.value() / 100.0
                    # 设置图像到预测器
                    self.sam_before.predictor.set_image(self.current_image_before)
                
                # 执行分割
                self.statusBar().showMessage(f"正在分割建筑物 ({x}, {y})...")
                QApplication.processEvents()
                
                building = self.sam_before.segment_building(x, y)
                
                if building:
                    # 添加到视图
                    self.left_view.add_building(building)
                    # 更新统计信息
                    self.update_stats()
                    self.statusBar().showMessage(f"✅ 已分割建筑物 {building['id']} - 面积: {building['area']}px²")
                else:
                    self.statusBar().showMessage("❌ 分割失败，请尝试点击建筑物的不同位置")
            elif image_type == "after":
                # 加载后时序图像到SAM
                if not hasattr(self, 'sam_after') or self.sam_after is None or self.sam_after.image_path != self.image_path_after:
                    # 直接使用现有的SAM预测器，不需要创建新的PixelPrecisionSAM实例
                    self.sam_after = PixelPrecisionSAM()
                    self.sam_after.predictor = self.sam.predictor
                    # 直接设置图像，不调用set_image方法避免重新加载模型
                    self.sam_after.image = self.current_image_after
                    self.sam_after.image_path = self.image_path_after
                    # 重置标注数据
                    self.sam_after.buildings = []
                    self.sam_after.current_building_id = 0
                    # 设置参数
                    self.sam_after.set_precision_level(self.precision_level)
                    self.sam_after.prediction_confidence = self.confidence_slider.value() / 100.0
                    # 设置图像到预测器
                    self.sam_after.predictor.set_image(self.current_image_after)
                
                # 执行分割
                self.statusBar().showMessage(f"正在分割建筑物 ({x}, {y})...")
                QApplication.processEvents()
                
                building = self.sam_after.segment_building(x, y)
                
                if building:
                    # 添加到视图
                    self.right_view.add_building(building)
                    # 更新统计信息
                    self.update_stats()
                    self.statusBar().showMessage(f"✅ 已分割建筑物 {building['id']} - 面积: {building['area']}px²")
                else:
                    self.statusBar().showMessage("❌ 分割失败，请尝试点击建筑物的不同位置")
    
    def on_building_selected(self, building_id, image_type):
        """建筑物被选中"""
        # 查找建筑物数据
        if building_id >= 0:  # SAM分割的建筑物
            building = None
            if image_type == "before" and hasattr(self, 'sam_before'):
                for b in self.sam_before.buildings:
                    if b["id"] == building_id:
                        building = b
                        break
            elif image_type == "after" and hasattr(self, 'sam_after'):
                for b in self.sam_after.buildings:
                    if b["id"] == building_id:
                        building = b
                        break
            
            if building:
                building_type_name = "先时序" if image_type == "before" else "后时序"
                self.update_building_info(building, image_type, building_type_name)
                # 保存选中状态
                self.current_selected_building = {
                    "id": building_id,
                    "type": "sam",
                    "image_type": image_type
                }
                # 启用删除按钮
                self.delete_btn.setEnabled(True)
            else:
                building_type_name = "先时序" if image_type == "before" else "后时序"
                self.update_building_info(None, image_type, building_type_name)
                # 清除选中状态
                self.current_selected_building = None
                self.delete_btn.setEnabled(False)
        else:  # 手动绘制的建筑物
            # 从视图项中获取多边形的点
            view = self.left_view if image_type == "before" else self.right_view
            if building_id in view.building_items:
                item = view.building_items[building_id]
                points = item.get_points()
                if len(points) > 2:
                    # 计算面积
                    points_array = np.array(points, dtype=np.int32)
                    area = cv2.contourArea(points_array)
                    
                    # 创建一个临时的建筑物信息字典
                    manual_building = {
                        'id': building_id,
                        'area': int(area),
                        'score': 0,
                        'points': points,
                        'vertices': len(points)
                    }
                    building_type_name = "先时序" if image_type == "before" else "后时序"
                    self.update_building_info(manual_building, image_type, f"{building_type_name}手动绘制")
                    # 保存选中状态
                    self.current_selected_building = {
                        "id": building_id,
                        "type": "manual",
                        "image_type": image_type
                    }
                    # 启用删除按钮
                    self.delete_btn.setEnabled(True)
                else:
                    building_type_name = "先时序" if image_type == "before" else "后时序"
                    self.update_building_info(None, image_type, building_type_name)
                    # 清除选中状态
                    self.current_selected_building = None
                    self.delete_btn.setEnabled(False)
            else:
                building_type_name = "先时序" if image_type == "before" else "后时序"
                self.update_building_info(None, image_type, building_type_name)
                # 清除选中状态
                self.current_selected_building = None
                self.delete_btn.setEnabled(False)
    
    def on_polygon_modified(self, building_id, points, image_type):
        """多边形修改事件"""
        print(f"{image_type}多边形 {building_id} 被修改，顶点数: {len(points)}")
        
        # 在修改完成后保存状态
        self.save_state(image_type)
        
        # 同步到自由标注窗口
        for dialog in self.free_annotation_dialogs:
            if dialog.image_type == image_type:
                dialog.polygon_updated.emit(building_id, points, image_type)
        
        self.update_stats()
    
    def update_building_info(self, building, image_type="before", building_type_name="先时序"):
        """更新建筑物信息显示"""
        if building:
            self.building_id_label.setText(f"{building['id']}")
            self.building_area_label.setText(f"{building['area']} px²")
            
            if building_type_name.endswith("手动绘制"):
                self.building_score_label.setText("-")
                self.vertices_label.setText(f"{building.get('vertices', len(building.get('points', [])))}")
            else:
                self.building_score_label.setText(f"{building.get('score', 0):.3f}")
                self.vertices_label.setText(f"{len(building.get('points', []))}")
            
            self.building_type_label.setText(building_type_name)
        else:
            self.building_id_label.setText("-")
            self.building_area_label.setText("- px²")
            self.building_score_label.setText("-")
            self.vertices_label.setText("-")
            self.building_type_label.setText(building_type_name)
    
    def update_stats(self):
        """更新统计信息"""
        before_count = len(self.sam_before.buildings) if hasattr(self, 'sam_before') and self.sam_before else 0
        after_count = len(self.sam_after.buildings) if hasattr(self, 'sam_after') and self.sam_after else 0
        
        # 计算手动绘制的建筑物
        before_manual = len([bid for bid in self.left_view.building_items.keys() if bid < 0])
        after_manual = len([bid for bid in self.right_view.building_items.keys() if bid < 0])
        
        self.stats_label.setText(f"先时序标注: {before_count + before_manual} | 后时序标注: {after_count + after_manual}")
    
    def update_confidence(self, value):
        """更新置信度阈值"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        
        # 更新两个SAM实例的置信度阈值
        if hasattr(self, 'sam_before') and self.sam_before:
            self.sam_before.prediction_confidence = confidence
        if hasattr(self, 'sam_after') and self.sam_after:
            self.sam_after.prediction_confidence = confidence
        
        self.statusBar().showMessage(f"置信度阈值设置为: {confidence:.2f}")
    
    def export_binary_mask(self):
        """导出二值图（基于先时序图像）"""
        if not hasattr(self, 'sam_before') or self.current_image_before is None:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("警告")
            msg_box.setText("请先加载先时序图像")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
            msg_box.exec_()
            return
        
        # 计算建筑物数量
        before_count = len(self.sam_before.buildings) if self.sam_before else 0
        before_manual = len([bid for bid in self.left_view.building_items.keys() if bid < 0])
        
        if before_count + before_manual == 0:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("警告")
            msg_box.setText("没有可导出的建筑物标注")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
            msg_box.exec_()
            return
        
        # 获取源图片的格式
        source_format = Path(self.image_path_before).suffix.lower()
        
        # 设置初始保存路径 - 使用与先时序图片相同的文件名
        initial_path = self.last_export_folder
        if initial_path and Path(initial_path).is_dir():
            initial_path = Path(initial_path) / Path(self.image_path_before).name
        else:
            initial_path = Path(self.image_path_before).name
        
        # 设置文件对话框过滤器
        if source_format == '.png':
            file_filter = "PNG图像 (*.png)"
        elif source_format == '.jpg' or source_format == '.jpeg':
            file_filter = "JPEG图像 (*.jpg *.jpeg)"
        elif source_format == '.bmp':
            file_filter = "BMP图像 (*.bmp)"
        else:
            # 默认使用PNG格式
            file_filter = "PNG图像 (*.png)"
            source_format = '.png'
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存建筑物标签", 
            str(initial_path),
            file_filter
        )
        
        if save_path:
            try:
                # 创建一个新的二值图
                h, w = self.current_image_before.shape[:2]
                binary_mask = np.zeros((h, w), dtype=np.uint8)
                
                # 绘制SAM分割的建筑物
                if self.sam_before:
                    for building in self.sam_before.buildings:
                        points = np.array(building["points"], dtype=np.int32)
                        if len(points) > 2:
                            # 裁剪坐标到图像范围内
                            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                            points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                            cv2.fillPoly(binary_mask, [points], 255)
                
                # 绘制手动绘制的建筑物
                for building_id, item in self.left_view.building_items.items():
                    if building_id < 0:  # 手动绘制的建筑物
                        points = np.array(item.get_points(), dtype=np.int32)
                        if len(points) > 2:
                            # 裁剪坐标到图像范围内
                            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                            points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                            cv2.fillPoly(binary_mask, [points], 255)
                
                # 将单通道灰度图转换为RGB三通道图像
                binary_mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
                
                # 确保保存格式与源图片相同
                if save_path.lower().endswith(source_format):
                    # 如果用户已经选择了正确的格式，直接保存
                    final_path = save_path
                else:
                    # 否则添加正确的扩展名
                    final_path = save_path + source_format
                
                # 根据文件格式设置保存参数，确保无损压缩
                if source_format == '.png':
                    # PNG压缩方式（采用Photoshop的压缩算法）
                    # 1. 差分预测过滤（自动选择最优类型）
                    # 2. DEFLATE压缩（LZ77 + 哈夫曼编码，压缩级别6）
                    # 3. 元数据清理（仅保留必要信息）
                    cv2.imwrite(final_path, binary_mask_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                elif source_format in ['.jpg', '.jpeg']:
                    # JPEG最大质量（接近无损）
                    cv2.imwrite(final_path, binary_mask_rgb, [cv2.IMWRITE_JPEG_QUALITY, 100])
                else:
                    # 其他格式使用默认无损设置
                    cv2.imwrite(final_path, binary_mask_rgb)
                # 更新上次导出的文件夹路径
                self.last_export_folder = str(Path(final_path).parent)
                
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("成功")
                msg_box.setText(f"✅ 二值图已保存到:\n{final_path}\n"
                    f"建筑物总数: {before_count + before_manual}\n"
                    f"图像尺寸: {self.current_image_before.shape[1]}×{self.current_image_before.shape[0]}\n"
                    f"颜色模式: RGB (3通道)\n"
                    f"文件格式: {source_format[1:].upper()}")
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; } QPushButton { min-width: 80px; }")
                msg_box.exec_()
                
                self.statusBar().showMessage(f"已导出二值图: {final_path}")
            except Exception as e:
                msg_box = QMessageBox()
                msg_box.setWindowTitle("错误")
                msg_box.setText(f"导出失败:\n{str(e)}")
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
                msg_box.exec_()
    
    def clear_all_annotations(self):
        """清除所有标注"""
        msg_box = QMessageBox()
        msg_box.setWindowTitle("确认")
        msg_box.setText("确定要清除所有标注吗？\n这将清除两个时序图像上的所有标注")
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        msg_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }")
        reply = msg_box.exec_()
        
        if reply == QMessageBox.Yes:
            # 在清除前保存状态
            if hasattr(self, 'sam_before') and self.sam_before:
                self.save_state("before")
            if hasattr(self, 'sam_after') and self.sam_after:
                self.save_state("after")
            
            # 清除先时序标注
            if hasattr(self, 'sam_before') and self.sam_before:
                self.sam_before.clear_all_buildings()
            self.left_view.clear_all_buildings()
            
            # 清除后时序标注
            if hasattr(self, 'sam_after') and self.sam_after:
                self.sam_after.clear_all_buildings()
            self.right_view.clear_all_buildings()
            
            # 同步到所有打开的自由标注窗口
            for dialog in self.free_annotation_dialogs:
                dialog.sync_to_main_window()
            
            # 重置信息显示
            self.building_id_label.setText("-")
            self.building_area_label.setText("- px²")
            self.building_score_label.setText("-")
            self.vertices_label.setText("-")
            self.building_type_label.setText("-")
            self.stats_label.setText("先时序标注: 0 | 后时序标注: 0")
            
            self.statusBar().showMessage("已清除所有建筑物标注")
    
    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key_F1:
            self.show_help()
        elif event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            # Ctrl+Z 撤销
            self.undo_last_action()
        elif event.key() == Qt.Key_P:
            # P键配合+/-调整精细度
            if event.modifiers() & Qt.ControlModifier:
                if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                    # Ctrl+P+增加精细度
                    if self.precision_level < 10:
                        self.precision_level += 1
                        self.precision_slider.setValue(self.precision_level)
                        self.update_precision(self.precision_level)
                elif event.key() == Qt.Key_Minus:
                    # Ctrl+P+减少精细度
                    if self.precision_level > 1:
                        self.precision_level -= 1
                        self.precision_slider.setValue(self.precision_level)
                        self.update_precision(self.precision_level)
        
        super().keyPressEvent(event)




class EasterEggDialog(QDialog):
    """彩蛋对话框，显示像素复活节蛋"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("彩蛋！")
        self.setModal(True)
        self.setFixedSize(600, 500)
        
        # 设置窗口样式
        self.setStyleSheet(""".QDialog {
            background-color: #f0f0f0;
            border: 2px solid #333333;
        }
        
        QLabel {
            font-family: Arial, sans-serif;
            font-size: 24px;
            color: #333333;
            text-align: center;
        }""")
        
        # 创建垂直布局
        layout = QVBoxLayout(self)
        
        # 创建图片标签用于显示彩蛋图片
        self.egg_label = QLabel()
        self.egg_label.setFixedSize(500, 400)
        self.egg_label.setAlignment(Qt.AlignCenter)
        
        # 加载22.jpg图片
        try:
            pixmap = QPixmap(resource_path("22.jpg"))
            if pixmap.isNull():
                raise Exception("无法加载图片")
            
            # 调整图片大小以适应标签，保持比例
            scaled_pixmap = pixmap.scaled(self.egg_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.egg_label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.egg_label.setText(f"无法加载彩蛋图片: {str(e)}")
        
        # 创建文字标签
        self.text_label = QLabel("恭喜你发现彩蛋，本软件由302cs倾情巨献！\n凭此彩蛋可到302兑换呼伦贝尔新鲜空气~")
        
        # 添加控件到布局
        layout.addWidget(self.egg_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.text_label, alignment=Qt.AlignCenter)
        layout.setSpacing(20)
        layout.setContentsMargins(10, 10, 10, 10)
    
    def paintEvent(self, event):
        # 不再需要绘制像素蛋，使用图片代替
        pass

    def draw_pixel_egg(self, painter, x, y):
        # 不再需要这个方法，使用图片代替
        pass
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置全局字体为微软雅黑，增大字体大小
    font = QFont()
    font.setFamily("Microsoft YaHei UI")  # 设置为微软雅黑
    font.setPointSize(14)
    app.setFont(font)
    
    # 显示主窗口
    window = SAMBuildingAnnotator()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()