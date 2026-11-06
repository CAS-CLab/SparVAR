def get_encode_decode_func(dynamic_scale_schedule):
    if dynamic_scale_schedule != 'infinity_elegant_clip20frames_v2':
        raise NotImplementedError(
            f'Unsupported release dynamic_scale_schedule: {dynamic_scale_schedule}; '
            'use infinity_elegant_clip20frames_v2'
        )
    from models.schedules.infinity_elegant import (
        video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info,
    )
    return video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info
