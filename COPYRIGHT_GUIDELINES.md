# Copyright Notice Guidelines for Open Source Projects

## 为什么需要版权声明？

### 1. **法律保护**
- 明确代码的所有权
- 保护你的知识产权
- 防止未经授权的使用

### 2. **开源合规**
- 符合开源许可证要求
- 尊重原作者的贡献
- 明确衍生作品的关系

### 3. **学术诚信**
- 正确归属原始工作
- 标注改进和扩展
- 便于引用和追溯

## 版权声明的标准格式

### 基本格式
```python
# Copyright (c) [Year] [Author/Organization]. [Description]
```

### 完整格式（推荐用于开源项目）
```python
# Copyright (c) [Year] [Original Author]. Original [Project Name].
# Copyright (c) [Year] [Your Name]. Modifications for [Your Project].
#
# Based on [Original Project] by [Original Author]
# Original repository: [URL]
# 
# Licensed under [License] - see LICENSE file for details.
```

## 针对你的项目：Efficient-LVSM

### 情况分析
你的项目是基于 **LVSM (ICLR 2025)** 开发的改进版本，应该：

1. ✅ **保留原作者版权**
2. ✅ **添加你的版权声明**
3. ✅ **说明是衍生作品**
4. ✅ **引用原始项目**

### 推荐的版权声明模板

#### 对于修改过的原始文件（如 `model/incremental.py`）

```python
# Copyright (c) 2025 Haian Jin. Original LVSM implementation (ICLR 2025).
# Copyright (c) 2025 [Your Name/Team]. Modifications for Efficient-LVSM.
#
# This code is based on the LVSM project by Haian Jin et al.
# Original paper: "LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias"
# Original repository: https://github.com/[original-repo-if-available]
# 
# Licensed under [Same as original or compatible license].
```

#### 对于全新创建的文件（如 `inference_incremental.py`）

```python
# Copyright (c) 2025 [Your Name/Team]. Efficient-LVSM project.
# 
# New incremental inference implementation extending LVSM framework.
# Built upon LVSM by Haian Jin et al. (ICLR 2025).
# 
# Licensed under [Your chosen license].
```

#### 对于配置文件（如 `model/repa_config.py`）

你已经正确修改为：
```python
# Copyright (c) 2025 Yihang Sun. Created for the Efficient LVSM project (ICLR 2026).
```

如果这个文件来自原项目，应该改为：
```python
# Copyright (c) 2025 Haian Jin. Original LVSM REPA configuration.
# Copyright (c) 2025 Yihang Sun. Extended configurations for Efficient-LVSM (ICLR 2026).
```

## 实际使用建议

### 方案 A：保守方式（推荐用于学术项目）
```python
# Copyright (c) 2025 Haian Jin et al. Original LVSM implementation (ICLR 2025).
# Copyright (c) 2025 [Your Name/Team]. Efficient-LVSM modifications and extensions.
#
# This work builds upon LVSM: "A Large View Synthesis Model with Minimal 3D Inductive Bias"
# by Haian Jin, Hanwen Jiang, Hao Tan, et al., ICLR 2025.
# 
# Major contributions in this version:
# - Incremental inference with KV caching
# - Memory-efficient view processing
# - Performance optimizations
#
# Licensed under [License Name].
```

### 方案 B：简洁方式（适合内部开发）
```python
# Copyright (c) 2025 [Your Name]. Efficient-LVSM.
# Based on LVSM by Haian Jin et al. (ICLR 2025).
```

### 方案 C：详细方式（适合商业项目）
```python
# Copyright (c) 2025 Haian Jin and contributors. Original LVSM implementation.
# Copyright (c) 2025 [Your Organization]. All modifications and enhancements.
#
# SPDX-License-Identifier: [License-Identifier]
#
# This file is part of Efficient-LVSM, an extended version of LVSM
# Original LVSM: https://github.com/[repo] (ICLR 2025)
# 
# Modifications include:
# - [List key changes]
#
# For the original LVSM license, see ORIGINAL_LICENSE
# For modifications license, see LICENSE
```

## 不同文件类型的处理

### 1. **从原项目大量修改的文件**
```python
# Copyright (c) 2025 Haian Jin. Original implementation.
# Copyright (c) 2025 [Your Name]. Modifications for Efficient-LVSM.
```

### 2. **从原项目轻微修改的文件**
```python
# Copyright (c) 2025 Haian Jin. Original LVSM implementation (ICLR 2025).
# Modified by [Your Name] for Efficient-LVSM (2025).
```

### 3. **完全新创建的文件**
```python
# Copyright (c) 2025 [Your Name]. Efficient-LVSM project.
# Built upon LVSM framework by Haian Jin et al.
```

### 4. **测试和工具脚本**
```python
# Copyright (c) 2025 [Your Name]. Efficient-LVSM utilities.
```

## 许可证（License）说明

### 常见开源许可证

1. **MIT License** - 最宽松，允许商业使用
2. **Apache 2.0** - 包含专利授权
3. **GPL v3** - 要求衍生作品也开源
4. **BSD 3-Clause** - 类似MIT但有额外条款
5. **Creative Commons** - 适合学术/文档

### 如何选择？

查看原项目（LVSM）的许可证：
```bash
# 在原项目仓库中查找
cat LICENSE
# 或
cat LICENSE.md
```

**原则**：
- ✅ 你的许可证必须**兼容**原项目许可证
- ✅ 如果原项目是MIT，你可以使用MIT、Apache 2.0等
- ✅ 如果原项目是GPL，你的衍生项目也必须是GPL

## 完整示例

### 文件头模板（推荐）

```python
# Copyright (c) 2025 Haian Jin et al. Original LVSM implementation (ICLR 2025).
# Copyright (c) 2025 [Your Name/Organization]. Efficient-LVSM modifications.
#
# This file is part of Efficient-LVSM, which extends the LVSM framework with:
# - Incremental inference capabilities
# - Optimized memory usage
# - Enhanced performance features
#
# Original LVSM: "A Large View Synthesis Model with Minimal 3D Inductive Bias"
# by Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang,
# Fujun Luan, Noah Snavely, and Zexiang Xu. ICLR 2025.
#
# Licensed under the [Same/Compatible License as original].
# See LICENSE file for full license text.
```

### LICENSE 文件示例

在项目根目录创建 `LICENSE` 文件：

```text
MIT License (or whatever license you choose)

Copyright (c) 2025 Haian Jin et al. - Original LVSM implementation
Copyright (c) 2025 [Your Name] - Efficient-LVSM modifications

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
[完整许可证文本]
```

## 特殊情况处理

### 1. **如果原项目没有明确许可证**
- 联系原作者获取许可
- 在论文/仓库中查找说明
- 保守起见，假设所有权归原作者

### 2. **多个贡献者**
```python
# Copyright (c) 2025 Haian Jin et al. - Original LVSM
# Copyright (c) 2025 [Contributor 1] - [Specific contribution]
# Copyright (c) 2025 [Contributor 2] - [Specific contribution]
```

### 3. **学术论文相关**
如果你要发表论文（如ICLR 2026），在代码中引用：
```python
# Copyright (c) 2025 [Your Name]. 
# Efficient-LVSM: [Paper Title] (ICLR 2026).
# 
# Based on LVSM by Haian Jin et al. (ICLR 2025).
```

## 检查清单

发布前确保：
- [ ] 所有源文件都有版权声明
- [ ] 正确归属原作者
- [ ] 明确标注你的贡献
- [ ] 包含许可证引用
- [ ] LICENSE 文件存在且正确
- [ ] README 中说明项目关系
- [ ] 引用原始论文/项目

## 推荐做法

### 对于你的 Efficient-LVSM 项目：

1. **在 README.md 中添加致谢**：
```markdown
## Acknowledgments

This project builds upon the excellent work of LVSM by Haian Jin et al. (ICLR 2025).

**Original LVSM**:
- Paper: "LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias"
- Authors: Haian Jin, Hanwen Jiang, Hao Tan, et al.
- Conference: ICLR 2025
- Repository: [link if available]

**Our Contributions**:
- Incremental inference with KV caching
- Memory optimization techniques
- Performance enhancements for multi-view scenarios
```

2. **创建 ORIGINAL_LICENSE 文件**（如果原项目有许可证）

3. **在论文中引用原作**

## 总结

✅ **必须做**：
- 保留原作者版权声明
- 添加你自己的版权声明
- 说明基于原项目
- 包含许可证信息

❌ **不要做**：
- 删除原作者版权
- 声称完全原创
- 使用不兼容的许可证
- 忽略原作者贡献

🎯 **最佳实践**：
简洁、清晰、诚实地表明代码来源和你的贡献。

---

**针对你的具体情况**，建议使用：

```python
# Copyright (c) 2025 Haian Jin et al. Original LVSM (ICLR 2025).
# Copyright (c) 2025 [Your Name/Team]. Efficient-LVSM modifications.
#
# Based on LVSM: https://github.com/[original-repo]
# Licensed under [License Name].
```

简洁、专业、合规！

