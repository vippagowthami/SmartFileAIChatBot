import os
import time
import uuid
import json
from pathlib import Path
from moviepy.editor import TextClip, ColorClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont

class VideoRenderer:
    """Renders an MP4 video from a script JSON using MoviePy."""
    
    def __init__(self, output_dir: str = "./data/rendered_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def render_script(self, script: dict, lang_code: str = "en") -> str:
        """
        Takes a script dict and renders an MP4.
        Returns the path to the rendered video.
        """
        job_id = str(uuid.uuid4())[:8]
        scene_clips = []
        
        scenes = script.get("scenes", [])
        if not scenes:
            raise ValueError("No scenes found in script")

        print(f"[VideoRenderer] Starting render for job {job_id} ({len(scenes)} scenes)")

        try:
            for i, scene in enumerate(scenes):
                print(f"[VideoRenderer] Processing scene {i+1}...")
                
                # 1. Generate Audio for Narration
                narration = scene.get("narration", "")
                audio_path = self.temp_dir / f"{job_id}_scene_{i}.mp3"
                
                tts = gTTS(text=narration, lang=lang_code)
                tts.save(str(audio_path))
                
                audio_clip = AudioFileClip(str(audio_path))
                duration = audio_clip.duration
                
                # 2. Create Visual Clip (Color background + Text overlay)
                # We use a dark gradient style background
                bg_clip = ColorClip(size=(1280, 720), color=(20, 25, 35), duration=duration)
                
                # Scene Title / Description text
                title_text = scene.get("scene_description", f"Scene {i+1}")
                if len(title_text) > 80:
                    title_text = title_text[:77] + "..."
                
                txt_clip = TextClip(
                    title_text,
                    fontsize=50,
                    color='white',
                    font='Arial-Bold',
                    size=(1000, None),
                    method='caption'
                ).set_duration(duration).set_position('center')
                
                # Visual Prompt (smaller at bottom)
                prompt_text = scene.get("visual_prompt", "")
                if prompt_text:
                    if len(prompt_text) > 120:
                        prompt_text = prompt_text[:117] + "..."
                    prompt_clip = TextClip(
                        f"Visual: {prompt_text}",
                        fontsize=24,
                        color='gray',
                        font='Arial',
                        size=(1100, None),
                        method='caption'
                    ).set_duration(duration).set_position(('center', 600))
                    
                    video_scene = CompositeVideoClip([bg_clip, txt_clip, prompt_clip])
                else:
                    video_scene = CompositeVideoClip([bg_clip, txt_clip])
                
                video_scene = video_scene.set_audio(audio_clip)
                scene_clips.append(video_scene)

            # 3. Concatenate and Write
            final_video = concatenate_videoclips(scene_clips, method="compose")
            output_filename = f"video_{job_id}.mp4"
            output_path = self.output_dir / output_filename
            
            print(f"[VideoRenderer] Exporting final video to {output_path}...")
            final_video.write_videofile(str(output_path), fps=24, codec="libx264", audio_codec="aac")
            
            return str(output_path)

        except Exception as e:
            print(f"[VideoRenderer Error] {e}")
            raise e
        finally:
            # Cleanup temp files would go here in production
            pass

def get_video_renderer():
    return VideoRenderer()
