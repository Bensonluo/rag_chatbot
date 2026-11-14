import type { Metadata } from "next";
import "./globals.css";
import { LanguageProvider } from "@/lib/i18n/context";

export const metadata: Metadata = {
  title: "Agent Store — 可托付的购物 Agent · 智能客服",
  description:
    "既是把事办完的购物 Agent，也是可溯源的智能客服。报价确认、幂等下单、断线不重复扣款、exactly-once 降价提醒。模拟商城，不产生真实交易。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning data-scroll-behavior="smooth">
      <body className="font-sans antialiased">
        <LanguageProvider>{children}</LanguageProvider>
      </body>
    </html>
  );
}
