// AntD 主题：仿 RAGFlow 的深蓝主色
// RAGFlow 官网主色是 #1e3a5f / #2a5298 那一档深蓝，我们微调一下让 AntD 默认色
// 配上去不冲突。
import { theme } from "antd";

export const ragTheme = {
  algorithm: theme.defaultAlgorithm,
  token: {
    colorPrimary: "#2a5298",   // 主色：深蓝（RAGFlow 风格）
    colorInfo:    "#2a5298",
    borderRadius: 6,
    fontSize: 14,
  },
  components: {
    Layout: {
      headerBg: "linear-gradient(135deg, #1e3a5f 0%, #2a5298 100%)",
      siderBg:  "#ffffff",
      bodyBg:   "#f5f7fa",
    },
    Menu: {
      itemSelectedBg: "#e6f0fa",
      itemSelectedColor: "#2a5298",
    },
  },
};
