"""Tests for the <think> reasoning-span stream filter."""

from app.services.llm.reasoning_filter import ReasoningFilter, strip_reasoning


def _feed_all(rf: ReasoningFilter, text: str, split: int) -> str:
    emitted: list[str] = []
    for i in range(0, len(text), split):
        out = rf.feed(text[i : i + split])
        assert isinstance(out, str)
        emitted.append(out)
    return "".join(emitted) + rf.flush()


class TestReasoningFilter:
    def test_plain_text_passes_through(self):
        rf = ReasoningFilter()
        assert rf.feed("您好，退款将在 7 个工作日内到账。") == "您好，退款将在 7 个工作日内到账。"
        assert rf.flush() == ""

    def test_leading_think_block_is_dropped(self):
        rf = ReasoningFilter()
        assert rf.feed("<think>用户问的是退货政策。</think>") == ""
        assert rf.feed("签收后 7 天内可退货。") == "签收后 7 天内可退货。"
        assert rf.flush() == ""

    def test_text_before_and_after_span(self):
        rf = ReasoningFilter()
        assert rf.feed("前缀。<think>推理</think>后缀。") == "前缀。后缀。"

    def test_tag_split_across_chunks_at_every_offset(self):
        text = "<think>推理内容</think>正式回答"
        for split in range(1, len(text) + 1):
            rf = ReasoningFilter()
            assert _feed_all(rf, text, split) == "正式回答", f"split={split}"

    def test_unterminated_think_is_dropped_at_flush(self):
        rf = ReasoningFilter()
        assert rf.feed("<think>模型没说完") == ""
        assert rf.flush() == ""

    def test_stray_angle_bracket_is_released(self):
        rf = ReasoningFilter()
        assert rf.feed("价格 < 100 元") == "价格 < 100 元"
        assert rf.flush() == ""

    def test_dangling_partial_open_tag_released_on_flush(self):
        rf = ReasoningFilter()
        assert rf.feed("a") == "a"
        assert rf.feed("<thi") == ""
        assert rf.flush() == "<thi"

    def test_two_reasoning_spans(self):
        rf = ReasoningFilter()
        assert rf.feed("<think>一</think>A<think>二</think>B") == "AB"

    def test_strip_reasoning_complete_text(self):
        assert strip_reasoning("<think>x</think>答案") == "答案"
        assert strip_reasoning("无标记文本") == "无标记文本"

    def test_real_shaped_stream(self):
        """A realistic GLM-shaped stream: think span first, answer after."""
        stream = [
            "<think> 用户问的是关于智能客服机器人的主要功能。",
            "我应该介绍我能提供的服务功能。 </think>",
            "\n\n您好！作为平台的智能客服，",
            "我可以为您提供订单查询、售后服务等服务。",
        ]
        rf = ReasoningFilter()
        out = "".join(rf.feed(c) for c in stream) + rf.flush()
        assert "<think>" not in out
        assert "用户问的是" not in out
        assert out.strip().startswith("您好")
