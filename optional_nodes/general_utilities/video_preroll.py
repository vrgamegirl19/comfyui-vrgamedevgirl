def add_preroll_frames(frames_per_scene, chunk_index, preroll_frames=6):
    """
    Adds extra frames to the FRONT for non-first chunks.
    Returns:
        total_frames_to_generate,
        preroll_frames_to_trim
    """
    if chunk_index == 0:
        return frames_per_scene, 0

    return frames_per_scene + preroll_frames, preroll_frames
