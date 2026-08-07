# GitHub 开源部署指南

## 步骤 1：创建 GitHub 仓库

### 方式 A：Web 界面（推荐新手）

1. 登录 https://github.com
2. 点击右上角 **+** → **New repository**
3. 填写：
   - **Repository name**: `medical-distill`（或你想要的名字）
   - **Description**: `用 API 蒸馏医学题库生成 SFT 数据 + LoRA 训练`
   - **Public**: 公开（可选）
   - **Add a README file**: 不勾选（我们已有 README.md）
   - **Add .gitignore**: 选择 **Python**
   - **Choose a license**: 选择 **MIT License**
4. 点击 **Create repository**

### 方式 B：命令行

```bash
# 登录 GitHub CLI
gh auth login

# 创建仓库（自动在 GitHub 创建）
cd E:\skill\test
gh repo create medical-distill \
  --public \
  --description="Medical data distillation and LoRA training" \
  --source=. \
  --remote=origin \
  --push
```

---

## 步骤 2：初始化 Git 仓库

```bash
cd E:\skill\test

# 初始化 git
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: Medical data distillation pipeline"

# 添加远程仓库
git remote add origin https://github.com/yourusername/medical-distill.git

# 推送
git push -u origin main
```

---

## 步骤 3：检查提交内容

```bash
# 查看提交历史
git log --oneline

# 查看仓库状态
git status

# 查看大文件（确保没有意外提交）
git ls-files -z | xargs -0 ls -s | sort -n -r | head -20
```

---

## 步骤 4：仓库结构（推荐）

```
medical-distill/
├── .github/
│   └── workflows/
│       └── ci.yml              # 自动化测试（可选）
├── scripts/                    # 核心脚本
│   ├── api_distill.py
│   ├── medqa_to_input.py
│   ├── merge_all_data.py
│   ├── train_lora.py
│   ├── run_all.sh
│   └── run_train.sh
├── data/                       # 数据（建议上传到 HuggingFace Hub）
│   └── README.md               # 数据获取说明
├── output/                     # 产出（不提交，加入 .gitignore）
├── tests/                      # 测试（可选）
├── .env.example                # API Key 模板
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── SETUP.md                    # 本地/云端部署说明
```

---

## 步骤 5：数据上传（可选）

如果数据集较大，建议上传到 HuggingFace Hub：

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 登录
huggingface-cli login

# 上传数据
huggingface-cli upload yourusername/medical-distill-data data/sft/ --repo-type dataset
```

---

## 步骤 6：后续维护

### 更新代码

```bash
# 修改后提交
git add .
git commit -m "Update: merge_all_data.py fix answer extraction"
git push
```

### 创建 Release

```bash
git tag v1.0.0
git push origin v1.0.0
# 然后在 GitHub 页面创建 Release
```

### Issues 和 PRs

- 设置 `Settings` → `Issues` 为开启
- 添加模板：`.github/ISSUE_TEMPLATE/`
- 添加 PR 模板：`.github/PULL_REQUEST_TEMPLATE.md`

---

## 注意事项

1. **不要提交 .env**：已加入 .gitignore
2. **不要提交大文件**：输出数据建议上传 HuggingFace
3. **README 要清晰**：让其他人能看懂怎么用
4. **添加 License**：保护你的知识产权
5. **保持 .gitignore 更新**：排除临时文件

---

## 快速检查清单

- [ ] .env 已排除
- [ ] output/ 已排除
- [ ] data/sft/ 已排除（或上传到 Hub）
- [ ] README.md 清晰完整
- [ ] LICENSE 已添加
- [ ] requirements.txt 完整
- [ ] 首次提交成功
