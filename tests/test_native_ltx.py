from pathlib import Path

from ltx25.native_ltx import DURATION_HEAD, build_native_command
from ltx25.schemas import GenerateRequest


def test_t2a_native_command_carries_duration_and_audio_guidance(tmp_path: Path):
    request = GenerateRequest(
        mode="t2a",
        prompt="soft rain and distant thunder",
        num_frames=None,
        min_seconds=2,
        max_seconds=7,
        steps=24,
        fps=24,
        audio_guidance_scale=3.0,
        audio_stg_scale=1.5,
        audio_rescale_scale=0.7,
        audio_skip_step=1,
        audio_stg_blocks=[2, 7],
    )

    command = build_native_command(
        request,
        tmp_path,
        tmp_path / "out.wav",
        python_executable="python-native",
    )

    assert command[:3] == ["python-native", "-m", "ltx_pipelines.t2a_one_stage"]
    assert command[command.index("--duration-head-path") + 1] == str(DURATION_HEAD)
    auto_index = command.index("--auto-duration")
    assert command[auto_index + 1 : auto_index + 3] == ["2.0", "7.0"]
    assert "--num-frames" not in command
    assert command[command.index("--audio-cfg-guidance-scale") + 1] == "3.0"
    assert command[command.index("--audio-stg-guidance-scale") + 1] == "1.5"
    assert command[command.index("--audio-rescale-scale") + 1] == "0.7"
    assert command[command.index("--audio-skip-step") + 1] == "1"
    blocks = command.index("--audio-stg-blocks")
    assert command[blocks + 1 : blocks + 3] == ["2", "7"]


def test_keyframe_native_command_preserves_ordered_frame_indices(tmp_path: Path):
    first_id = "a" * 32
    last_id = "b" * 32
    (tmp_path / f"{first_id}.png").write_bytes(b"first")
    (tmp_path / f"{last_id}.png").write_bytes(b"last")

    request = GenerateRequest(
        mode="keyframe_interpolation",
        prompt="camera travels through the scene",
        width=768,
        height=512,
        num_frames=121,
        conditions=[
            {"asset_id": last_id, "kind": "image", "index": 1, "frame_index": 120},
            {"asset_id": first_id, "kind": "image", "index": 0, "frame_index": 0},
        ],
    )

    command = build_native_command(
        request,
        tmp_path,
        tmp_path / "out.mp4",
        python_executable="python-native",
    )

    assert command[:3] == ["python-native", "-m", "ltx_pipelines.keyframe_interpolation"]
    assert command[command.index("--num-frames") + 1] == "121"
    assert command[command.index("--width") + 1] == "768"
    assert command[command.index("--height") + 1] == "512"

    image_positions = [index for index, value in enumerate(command) if value == "--image"]
    assert len(image_positions) == 2
    first = command[image_positions[0] + 1 : image_positions[0] + 4]
    second = command[image_positions[1] + 1 : image_positions[1] + 4]
    assert first[0].endswith(f"{first_id}.png")
    assert first[1:] == ["0", "1.0"]
    assert second[0].endswith(f"{last_id}.png")
    assert second[1:] == ["120", "1.0"]


def test_dfr_native_command_exposes_spatial_and_temporal_rounds(tmp_path: Path):
    request = GenerateRequest(
        mode="dfr",
        prompt="production quality cinematic landscape",
        width=768,
        height=512,
        num_frames=121,
        dfr_spatial_upscalings=1,
        dfr_temporal_upscalings=2,

    )

    command = build_native_command(
        request,
        tmp_path,
        tmp_path / "out.mp4",
        python_executable="python-native",
    )

    assert command[:3] == ["python-native", "-m", "ltx_pipelines.dfr_pipeline"]
    assert "--detailing-lora" in command
    assert command[command.index("--spatial-upscalings") + 1] == "1"
    assert command[command.index("--temporal-upscalings") + 1] == "2"
    assert "--temporal-upsampler-path" in command
    assert "--distilled-lora" not in command
    assert "--hdr" not in command
