import type { Locale } from './context';

const translations = {
  zh: {
    nav: {
      home: '首页',
      roadmap: '路线图',
      oss: '开源贡献',
    },
    header: {
      lightMode: '亮色模式',
      darkMode: '暗色模式',
    },
    masthead: {
      roadmap: '路线图',
    },
    about: {
      title: '技术技能',
      description: '我在 ML 工程全栈中使用的核心技术与工具。',
    },
    projects: {
      all: '所有项目',
      liveDemo: '在线演示',
      interactiveDemo: '互动演示',
      viewDetails: '查看详情',
      techStack: '技术栈',
    },
    contact: {
      title: '联系我',
      description: '我目前正在寻找新机会。无论你有问题还是只想打个招呼，我都会尽力回复。',
      email: '邮箱',
      linkedin: 'LinkedIn',
      github: 'GitHub',
      linkedinValue: '在 LinkedIn 上联系',
      githubValue: '查看我的代码仓库',
    },
    footer: {
      allRightsReserved: '版权所有',
    },
    projectDetail: {
      backToList: '返回项目列表',
      viewSource: '查看源码',
      liveDemo: '在线演示',
      overview: '概述',
      techStack: '技术栈',
      keyFeatures: '核心特性',
    },
    growing: {
      phaseRoadmap: '阶段路线图',
      locked: '已锁定',
      viewing: '查看中',
      completed: '已完成',
      active: '进行中',
      upcoming: '待开始',
      tasks: '工作项',
      deliverables: '交付物',
      reflection: '复盘与判断',
      progress: '进度',
      phaseProgress: '交付进度',
      phases: '阶段',
      totalWeeks: '总周数',
      totalTasks: '总任务数',
      overallProgress: '路线图进度',
      phase3Locked: '阶段 3 已锁定',
      phase3LockedDesc: '请先完成阶段 0-2。职业转型内容在基础建设完成后解锁。',
    },
    voice: {
      browserNotSupported: '浏览器不支持语音识别',
      recognitionFailed: '语音识别未能启动，请重试',
      needsGoogle: '语音识别需要 Google 服务',
      micDenied: '麦克风权限被拒绝',
      clickToStop: '点击停止',
      clickToRecord: '点击开始录音',
      recognizing: '识别中...',
      autoMode: '自动模式',
      startPanel: '启动面板',
    },
    oss: {
      headline: '主要的开源贡献',
      statsProjects: '个项目',
      domainLabel: '贡献领域',
      statsLines: '行代码',
      statsPrs: '个 PR',
      statsMerged: '个已合并',
      colPrs: 'PR',
      footer: '数据来自 GitHub 公开 API',
    },
  },
  en: {
    nav: {
      home: 'Home',
      roadmap: 'Roadmap',
      oss: 'OSS',
    },
    header: {
      lightMode: 'Light Mode',
      darkMode: 'Dark Mode',
    },
    masthead: {
      roadmap: 'Roadmap',
    },
    about: {
      title: 'Technical Skills',
      description: 'Core technologies and tools I work with across the ML engineering stack.',
    },
    projects: {
      all: 'All Projects',
      liveDemo: 'Live Demo',
      interactiveDemo: 'Interactive Demo',
      viewDetails: 'View Details',
      techStack: 'Tech Stack',
    },
    contact: {
      title: 'Get In Touch',
      description: "I'm currently looking for new opportunities. Whether you have a question or just want to say hi, I'll try my best to get back to you.",
      email: 'Email',
      linkedin: 'LinkedIn',
      github: 'GitHub',
      linkedinValue: 'Connect on LinkedIn',
      githubValue: 'Check out my repositories',
    },
    footer: {
      allRightsReserved: 'All rights reserved.',
    },
    projectDetail: {
      backToList: 'Back to Projects',
      viewSource: 'View Source',
      liveDemo: 'Live Demo',
      overview: 'Overview',
      techStack: 'Tech Stack',
      keyFeatures: 'Key Features',
    },
    growing: {
      phaseRoadmap: 'Phase Roadmap',
      locked: 'Locked',
      viewing: 'Viewing',
      completed: 'Completed',
      active: 'In Progress',
      upcoming: 'Upcoming',
      tasks: 'Work Items',
      deliverables: 'Deliverables',
      reflection: 'Reflection & Decision',
      progress: 'Progress',
      phaseProgress: 'Delivery Progress',
      phases: 'Phases',
      totalWeeks: 'Total Weeks',
      totalTasks: 'Total Tasks',
      overallProgress: 'Roadmap Progress',
      phase3Locked: 'Phase 3 Locked',
      phase3LockedDesc: 'Complete Phase 0-2 first. Career transition content unlocks after building the foundation.',
    },
    voice: {
      browserNotSupported: 'Browser does not support speech recognition',
      recognitionFailed: 'Speech recognition could not start. Please try again.',
      needsGoogle: 'Speech recognition requires Google services',
      micDenied: 'Microphone permission denied',
      clickToStop: 'Click to stop',
      clickToRecord: 'Click to start recording',
      recognizing: 'Recognizing...',
      autoMode: 'Auto Mode',
      startPanel: 'Start Panel',
    },
    oss: {
      headline: 'Main open-source contributions',
      footer: 'Data via the GitHub public API',
      colPrs: 'PR',
      statsProjects: 'projects',
      domainLabel: 'Focus',
      statsLines: 'lines',
      statsPrs: 'PRs',
      statsMerged: 'merged',
    },
  },
} as const;

type NestedValue = string | { [key: string]: NestedValue };

export function getTranslation(locale: Locale, key: string): string {
  const keys = key.split('.');
  let result: NestedValue | undefined = translations[locale] as unknown as NestedValue;
  for (const k of keys) {
    if (typeof result === 'object' && result !== null) {
      result = result[k];
    } else {
      return key;
    }
  }
  return typeof result === 'string' ? result : key;
}

export { translations };
