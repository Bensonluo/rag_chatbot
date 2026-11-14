import type { Metadata } from "next";
import { StoreLanding } from "./store-landing";

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"
  ),
  title: "Agent Store — 可托付的购物 Agent · 智能客服",
  description:
    "既是把事办完的购物 Agent，也是可溯源的智能客服。购物：搜索 → 推荐 → 报价 → 你确认 → 下单 → 付款再确认；客服：订单时间线、带引用政策问答、幂等退款、人工衔接工单。报价确认、幂等下单、断线不重复扣款、exactly-once 降价提醒。模拟商城，不产生真实交易。",
  openGraph: {
    title: "Agent Store · 购物 Agent × 智能客服",
    description:
      "shop & support in one agent · quotes before orders · idempotent payments · cited policy answers · human handoff — 模拟商城演示，不产生真实交易",
    images: ["/images/projects/rag-cover.png"],
    type: "website",
  },
};

export default function StorePage() {
  return <StoreLanding />;
}
