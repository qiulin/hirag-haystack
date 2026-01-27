# HiRAG-Haystack 可视化功能

## 概述

已为 HiRAG-Haystack 项目添加了完整的交互式可视化功能，支持知识图谱、社区结构、查询路径和统计分析的可视化展示。

## 安装依赖

```bash
pip install -e ".[visualization]"
```

或者手动安装：

```bash
pip install pyvis>=0.3.2 plotly>=5.18.0 kaleido>=0.2.0
```

## 使用方式

### 方式 1: 通过 HiRAG 高层 API（推荐）

```python
from hirag_haystack import HiRAG

# 初始化 HiRAG
hirag = HiRAG(working_dir="./hirag_data")

# 生成所有可视化
results = hirag.visualize(kind="all")

# 生成特定可视化
hirag.visualize(kind="graph", layout="force", color_by="entity_type")
hirag.visualize(kind="communities", min_community_size=5)
hirag.visualize(kind="stats", show_top_n=20)
```

### 方式 2: 直接使用 GraphVisualizer 组件

```python
from hirag_haystack.components import GraphVisualizer
from hirag_haystack import HiRAG

hirag = HiRAG(working_dir="./hirag_data")
visualizer = GraphVisualizer(output_dir="./visualizations")

# 知识图谱可视化
kg_path = visualizer.visualize_knowledge_graph(
    graph_store=hirag.graph_store,
    layout="force",
    color_by="entity_type",
    show_labels=True,
    physics=True,
)

# 社区可视化
comm_path = visualizer.visualize_communities(
    communities=hirag.communities,
    graph_store=hirag.graph_store,
    min_community_size=3,
)

# 查询路径可视化
path = hirag.graph_store.shortest_path("ENTITY_A", "ENTITY_B")
path_path = visualizer.visualize_query_path(
    path=path,
    graph_store=hirag.graph_store,
    show_context=1,
)

# 统计图表
stats_path = visualizer.visualize_entity_stats(
    graph_store=hirag.graph_store,
    chart_types=["distribution", "degree", "overview"],
)
```

## 可视化类型

### 1. 知识图谱可视化

展示完整的实体关系网络。

**参数选项：**
- `layout`: 布局算法（"force", "hierarchical", "circular"）
- `color_by`: 着色方式（"entity_type", "community", "degree"）
- `node_size`: 节点大小（"degree", "constant"）
- `filter_min_degree`: 最小度数阈值
- `filter_max_nodes`: 最大节点数（用于大型图）
- `show_labels`: 是否显示标签
- `physics`: 是否启用物理模拟

**特性：**
- 交互式节点拖拽
- 缩放和平移
- 悬停显示实体详情
- 物理引擎模拟布局

### 2. 社区结构可视化

展示社区聚类结果。

**参数选项：**
- `min_community_size`: 最小社区大小
- `show_community_labels`: 显示社区级标签
- `show_entity_labels`: 显示实体级标签
- `include_descriptions`: 包含社区描述

**特性：**
- 不同颜色区分社区
- 社区内实体聚集
- 显示社区描述

### 3. 查询路径可视化

可视化跨社区检索路径。

**参数选项：**
- `path`: 实体路径列表
- `show_context`: 显示邻居节点层数
- `animate`: 是否动画
- `highlight_entities`: 额外高亮的实体

**特性：**
- 路径节点高亮（红色）
- 邻居上下文显示（灰色）
- 关系描述显示

### 4. 实体统计图表

展示提取统计信息。

**图表类型：**
- `distribution`: 实体类型分布（饼图）
- `degree`: Top 实体连接数（柱状图）
- `overview`: 图概览（指标卡片）

**特性：**
- 交互式图表
- 多图表组合
- 统计指标显示

## 运行示例

```bash
# 1. 首先创建示例数据
python examples/basic_usage.py

# 2. 生成可视化
python examples/visualizations.py
```

## 输出文件

所有可视化生成为独立的 HTML 文件，可在浏览器中打开：

- `knowledge_graph.html` - 知识图谱
- `communities.html` - 社区结构
- `query_path.html` - 查询路径
- `entity_stats.html` - 统计图表

默认保存位置：
- HiRAG API: `./hirag_data/visualizations/`
- GraphVisualizer: `./hirag_visualizations/`（可自定义）

## 自定义选项

### 颜色方案

实体类型颜色（在 `utils/color_utils.py` 中定义）：
- ORGANIZATION: #FF6B6B (红)
- PERSON: #4ECDC4 (青)
- LOCATION: #45B7D1 (蓝)
- PRODUCT: #FFA07A (橙)
- EVENT: #98D8C8 (绿)
- CONCEPT: #F7DC6F (黄)
- TECHNICAL_TERM: #BB8FCE (紫)
- UNKNOWN: #95A5A6 (灰)

### 高级用法

```python
# 自定义知识图谱
visualizer.visualize_knowledge_graph(
    graph_store=hirag.graph_store,
    layout="force",
    color_by="community",  # 按社区着色
    node_size="degree",    # 按度数调整大小
    filter_min_degree=2,   # 只显示度数≥2的节点
    filter_max_nodes=100,  # 最多100个节点
    show_labels=True,
    tooltip_fields=["entity_name", "entity_type", "description"],
    physics=True,
    height="1000px",
)
```

## 性能建议

对于大型知识图（>1000 节点）：
- 使用 `filter_max_nodes` 限制节点数
- 使用 `filter_min_degree` 过滤低连接节点
- 关闭 `physics` 或减少迭代次数
- 使用社区聚合视图

## 实现的文件

### 新建文件
- [hirag_haystack/utils/color_utils.py](hirag_haystack/utils/color_utils.py) - 颜色工具
- [hirag_haystack/components/graph_visualizer.py](hirag_haystack/components/graph_visualizer.py) - 主可视化组件
- [examples/visualizations.py](examples/visualizations.py) - 使用示例

### 修改文件
- [pyproject.toml](pyproject.toml) - 添加可视化依赖
- [hirag_haystack/components/__init__.py](hirag_haystack/components/__init__.py) - 导出 GraphVisualizer
- [hirag_haystack/__init__.py](hirag_haystack/__init__.py) - 添加 HiRAG.visualize() 方法

## 技术栈

- **pyvis**: 基于 vis.js 的交互式网络图
- **plotly**: 统计图表
- **kaleido**: 静态图片导出（可选）

## 后续改进建议

1. 添加更多布局算法（如 Sugiyama, stress）
2. 支持大型图的懒加载
3. 添加时间轴动画（如果有时间信息）
4. 支持 Neo4j Browser 风格的查询界面
5. 添加导出为 PNG/SVG 功能
6. 添加 3D 可视化支持

## 常见问题

**Q: 如何在 Jupyter Notebook 中显示可视化？**

```python
from hirag_haystack import HiRAG

hirag = HiRAG(working_dir="./hirag_data")
hirag.visualize(kind="graph")

# 在 Jupyter 中
from IPython.display import IFrame
IFrame(src="./hirag_data/visualizations/knowledge_graph.html", width=1000, height=800)
```

**Q: 如何自定义颜色？**

编辑 `utils/color_utils.py` 中的 `DEFAULT_TYPE_COLORS` 字典。

**Q: 图太大无法渲染？**

使用 `filter_max_nodes` 参数限制节点数，或使用社区视图。

**Q: 如何保存为静态图片？**

```python
import kaleido
fig.write_image("output.png")  # Plotly 图表
# pyvis 暂不支持直接导出，可在浏览器中截图
```
