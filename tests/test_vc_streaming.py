import pytest
import torch

from xvc2_codec.config import CodecConfig
from xvc2_codec.vc import StreamingLatentConverter, StreamingVC


def codec_config():
    return CodecConfig(
        student_dim=8,
        keep_trunk_dim=8,
        inv_dim=4,
        dyn_dim=2,
        edit_dim=4,
        encoder_channels=2,
        inv_projection_dim=4,
        dyn_projection_dim=2,
        edit_projection_dim=4,
        fusion_channels=8,
        fusion_kernel_size=3,
        fusion_dilations=(1, 2),
        decoder_channels=(16, 8, 4, 2),
        residual_kernels=(3, 5),
        residual_dilations=(1, 2),
        dropout=0,
    )


def converter():
    torch.manual_seed(7)
    model = StreamingLatentConverter(codec_config(), condition_dim=4, channels=8, dilations=(1, 2))
    # Exercise the temporal and condition paths, rather than testing only identity initialization.
    torch.nn.init.normal_(model.output.weight, std=0.1)
    return model.eval()


def latents(frames=19):
    torch.manual_seed(9)
    return (
        torch.randn(1, frames, 4),
        torch.randn(1, frames, 2),
        torch.randn(1, 4, frames),
        torch.randn(1, 4),
    )


def test_converter_matches_irregular_chunks_and_empty_flush():
    model = converter()
    inv, dyn, edit, condition = latents()
    with torch.no_grad():
        expected = model(inv, dyn, edit, condition)
    state = model.init_streaming_state(condition)
    parts = {key: [] for key in expected}
    start = 0
    for count in (0, 1, 4, 2, 7, 5, 0):
        output, state = model.forward_chunk(
            inv[:, start : start + count],
            dyn[:, start : start + count],
            edit[:, :, start : start + count],
            state,
            is_final=start == inv.shape[1],
        )
        start += count
        for key, values in parts.items():
            values.append(output[key])
    assert state.finalized
    for key, values in parts.items():
        torch.testing.assert_close(
            torch.cat(values, dim=2 if key == "z_edit" else 1),
            expected[key],
            rtol=1e-5,
            atol=1e-6,
        )
    with pytest.raises(RuntimeError, match="finalized"):
        model.forward_chunk(inv[:, :1], dyn[:, :1], edit[:, :, :1], state)


def test_converter_is_causal_conditioned_and_does_not_change_inv():
    model = converter()
    inv, dyn, edit, condition = latents()
    with torch.no_grad():
        baseline = model(inv, dyn, edit, condition)
        changed_inv, changed_dyn, changed_edit = inv.clone(), dyn.clone(), edit.clone()
        changed_inv[:, 8:] += 5
        changed_dyn[:, 8:] -= 3
        changed_edit[:, :, 8:] += 7
        changed = model(changed_inv, changed_dyn, changed_edit, condition)
        other_voice = model(inv, dyn, edit, -condition)
    assert baseline["z_inv"] is inv
    torch.testing.assert_close(baseline["z_dyn"][:, :8], changed["z_dyn"][:, :8])
    torch.testing.assert_close(baseline["z_edit"][:, :, :8], changed["z_edit"][:, :, :8])
    assert not torch.allclose(baseline["z_dyn"], other_voice["z_dyn"])
    assert not torch.allclose(baseline["z_edit"], other_voice["z_edit"])


def test_identity_initialization_and_reset():
    model = StreamingLatentConverter(codec_config(), condition_dim=4, channels=8).eval()
    inv, dyn, edit, condition = latents()
    result = model(inv, dyn, edit, condition)
    torch.testing.assert_close(result["z_dyn"], dyn)
    torch.testing.assert_close(result["z_edit"], edit)
    model = converter()
    first = model.init_streaming_state(condition)
    before, _ = model.forward_chunk(inv[:, :3], dyn[:, :3], edit[:, :, :3], first)
    model.forward_chunk(inv[:, 3:], dyn[:, 3:], edit[:, :, 3:], first, True)
    reset = model.init_streaming_state(condition)
    after, _ = model.forward_chunk(inv[:, :3], dyn[:, :3], edit[:, :, :3], reset)
    torch.testing.assert_close(before["z_edit"], after["z_edit"])
    assert not reset.finalized
    assert not torch.equal(first.blocks[0], reset.blocks[0])


def test_converter_rejects_mismatched_frames_and_training_stream():
    model = converter()
    inv, dyn, edit, condition = latents()
    with pytest.raises(ValueError, match="shapes"):
        model(inv, dyn[:, :-1], edit, condition)
    model.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        model.forward_chunk(inv, dyn, edit, model.init_streaming_state(condition))


def make_pipeline():
    student_module = pytest.importorskip("xvc2_student.model")
    from xvc2_student.config import ModelConfig

    from xvc2_codec.model import LargeStreamingCodec

    torch.manual_seed(13)
    student = student_module.StreamingPhoneEncoder(
        ModelConfig(
            conv_dim=(4,) * 7,
            model_dim=8,
            num_heads=2,
            ffn_dim=16,
            num_layers=2,
            segment_length=4,
            left_context_length=8,
            right_context_length=2,
            teacher_dim=8,
            dropout=0,
        )
    )
    return StreamingVC(student, LargeStreamingCodec(codec_config()), converter()).eval()


def stream(model, waveform, condition):
    state = model.init_streaming_state(condition)
    outputs = []
    position = 0
    sizes = (1, 319, 7, 400, 83, 641, 5)
    while position < waveform.shape[-1]:
        count = sizes[len(outputs) % len(sizes)]
        chunk = waveform[:, :, position : position + count]
        position += chunk.shape[-1]
        output, state = model.forward_chunk(chunk, state)
        outputs.append(output)
        assert state.waveform_buffer.shape[-1] < 320
        assert state.edit_buffer.shape[-1] <= 7
    tail, state = model.forward_chunk(waveform[:, :, :0], state, is_final=True)
    outputs.append(tail)
    return torch.cat(outputs, dim=-1), state


@pytest.mark.parametrize("samples", [400, 3200, 6400, 6479])
def test_raw_pipeline_matches_offline_and_trims_only_unsupported_tail(samples):
    model = make_pipeline()
    waveform, condition = torch.randn(1, 1, samples), torch.randn(1, 4)
    with torch.no_grad():
        offline = model(waveform, torch.tensor([samples]), condition)
    chunked, state = stream(model, waveform, condition)
    torch.testing.assert_close(chunked, offline["reconstruction"], rtol=1e-4, atol=2e-6)
    assert chunked.shape[-1] == ((samples - 400) // 320 + 1) * 320
    assert offline["output_lengths"].item() == chunked.shape[-1]
    assert state.finalized and state.student.finalized and state.converter.finalized
    assert state.waveform_buffer.shape[-1] == state.edit_buffer.shape[-1] == 0
    with pytest.raises(RuntimeError, match="finalized"):
        model.forward_chunk(waveform[:, :, :1], state)
    restarted, _ = stream(model, waveform, condition)
    torch.testing.assert_close(restarted, chunked)


@pytest.mark.parametrize("samples", [0, 100, 319, 399])
def test_raw_stream_shorter_than_student_receptive_field_emits_no_fake_audio(samples):
    model = make_pipeline()
    waveform, condition = torch.randn(1, 1, samples), torch.randn(1, 4)
    output, state = stream(model, waveform, condition)
    offline = model(waveform, torch.tensor([samples]), condition)
    assert output.shape == (1, 1, 0)
    assert offline["reconstruction"].shape == output.shape
    assert offline["output_lengths"].item() == 0
    assert state.finalized


def test_decoder_gradient_reaches_converter_but_frozen_backbones_stay_frozen():
    model = make_pipeline().train()
    assert model.converter.training
    assert not model.student.training and not model.codec.training
    condition = torch.randn(1, 4, requires_grad=True)
    output = model(torch.randn(1, 1, 3200), torch.tensor([3200]), condition)
    output["reconstruction"].square().mean().backward()
    assert model.converter.output.weight.grad.abs().sum() > 0
    assert model.converter.blocks[0].condition.weight.grad.abs().sum() > 0
    assert condition.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.codec.parameters())
    assert all(parameter.grad is None for parameter in model.student.parameters())


def test_interleaved_streams_have_independent_caches_and_condition_is_copied():
    model = make_pipeline()
    wave_a, wave_b = torch.randn(1, 1, 3200), torch.randn(1, 1, 4000)
    condition_a, condition_b = torch.randn(1, 4), torch.randn(1, 4)
    state_a, state_b = (
        model.init_streaming_state(condition_a),
        model.init_streaming_state(condition_b),
    )
    condition_a_copy = condition_a.clone()
    condition_a.zero_()
    torch.testing.assert_close(state_a.converter.condition, condition_a_copy)
    outputs = [[], []]
    for index in range(0, 4000, 317):
        for stream_index, (wave, state) in enumerate(((wave_a, state_a), (wave_b, state_b))):
            output, _ = model.forward_chunk(wave[:, :, index : index + 317], state)
            outputs[stream_index].append(output)
    for stream_index, (wave, condition, state) in enumerate(
        ((wave_a, condition_a_copy, state_a), (wave_b, condition_b, state_b))
    ):
        tail, _ = model.forward_chunk(wave[:, :, :0], state, True)
        outputs[stream_index].append(tail)
        alone, _ = stream(model, wave, condition)
        torch.testing.assert_close(torch.cat(outputs[stream_index], dim=-1), alone)


def test_pair_forward_uses_source_content_and_target_teacher_with_valid_lengths():
    model = make_pipeline().train()
    batch = {
        "source": {"waveform": torch.randn(2, 1, 3840), "frames": torch.tensor([12, 2])},
        "target": {"waveform": torch.randn(2, 1, 3840), "frames": torch.tensor([12, 2])},
    }
    batch["source"]["waveform"][1, :, 640:] = 0
    batch["target"]["waveform"][1, :, 640:] = 0
    result = model.forward_pair(batch, torch.randn(2, 4))
    source = model._encode(batch["source"]["waveform"], torch.tensor([3840, 640]))
    target = model._encode(batch["target"]["waveform"], torch.tensor([3840, 640]))
    torch.testing.assert_close(result["prediction"]["z_inv"], source["z_inv"])
    torch.testing.assert_close(result["target_latents"]["z_edit"], target["z_edit"])
    assert not result["target_latents"]["z_edit"].requires_grad
    assert result["prediction"]["reconstruction"].requires_grad
    assert result["frames"].tolist() == [11, 1]
    assert result["prediction"]["output_lengths"].tolist() == [3520, 320]
    assert result["target_waveform"].shape == (2, 1, 3520)
