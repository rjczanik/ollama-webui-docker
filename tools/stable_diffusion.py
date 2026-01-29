"""
title: Stable Diffusion Image Generator
author: rjczanik
version: 1.1.0
license: MIT
description: Generate AI images from text descriptions using Stable Diffusion. When the user asks to create, draw, generate, or make an image, picture, artwork, illustration, or photo - use this tool.
requirements: requests
"""

import requests
from typing import Optional
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        STABLE_DIFFUSION_URL: str = Field(
            default="http://stable-diffusion:17860",
            description="The URL of the Stable Diffusion WebUI API"
        )
        DEFAULT_STEPS: int = Field(
            default=20,
            description="Default number of sampling steps"
        )
        DEFAULT_CFG_SCALE: float = Field(
            default=7.0,
            description="Default CFG scale (guidance)"
        )
        DEFAULT_WIDTH: int = Field(
            default=512,
            description="Default image width"
        )
        DEFAULT_HEIGHT: int = Field(
            default=512,
            description="Default image height"
        )
        DEFAULT_SAMPLER: str = Field(
            default="Euler a",
            description="Default sampler method"
        )

    def __init__(self):
        self.valves = self.Valves()

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted, ugly, deformed, low resolution, pixelated",
    ) -> str:
        """
        Generate an AI image from a text description using Stable Diffusion.
        
        USE THIS TOOL WHEN THE USER ASKS TO:
        - Create, generate, draw, or make an image/picture/artwork/illustration/photo
        - Visualize something ("show me what X looks like")
        - Create visual content of any kind
        - "Draw me a...", "Make a picture of...", "Generate an image of..."
        
        PROMPT GUIDELINES - Create detailed, descriptive prompts:
        - Describe the subject clearly (person, animal, object, scene)
        - Include style keywords: "digital art", "oil painting", "photograph", "anime", "realistic", "3D render"
        - Add quality boosters: "highly detailed", "masterpiece", "professional", "8k resolution"
        - Specify lighting: "dramatic lighting", "soft light", "golden hour", "studio lighting"
        - Include mood/atmosphere: "peaceful", "epic", "mysterious", "vibrant"
        - Mention composition: "portrait", "landscape", "close-up", "wide angle"
        
        EXAMPLE GOOD PROMPTS:
        - "A majestic lion in the savanna at sunset, golden hour lighting, wildlife photography, highly detailed, National Geographic style"
        - "Cozy coffee shop interior, warm lighting, anime style, detailed background, peaceful atmosphere"
        - "Futuristic cyberpunk city at night, neon lights, rain, cinematic, highly detailed digital art"
        
        :param prompt: Detailed text description of the image to generate. Be specific and descriptive. Include style, mood, lighting, and quality keywords for best results.
        :param negative_prompt: Things to AVOID in the image. Default includes common quality issues. Add specific things the user doesn't want.
        :return: The generated image displayed in the chat, or an error message if generation fails.
        """
        
        width = self.valves.DEFAULT_WIDTH
        height = self.valves.DEFAULT_HEIGHT
        steps = self.valves.DEFAULT_STEPS
        cfg_scale = self.valves.DEFAULT_CFG_SCALE
        sampler_name = self.valves.DEFAULT_SAMPLER
        
        # Ensure dimensions are multiples of 64
        width = (width // 64) * 64
        height = (height // 64) * 64
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "seed": -1,
            "sampler_name": sampler_name,
            "batch_size": 1,
            "n_iter": 1,
        }
        
        try:
            api_url = f"{self.valves.STABLE_DIFFUSION_URL}/sdapi/v1/txt2img"
            response = requests.post(api_url, json=payload, timeout=600)
            response.raise_for_status()
            
            result = response.json()
            
            if "images" in result and len(result["images"]) > 0:
                image_base64 = result["images"][0]
                return f"![Generated Image](data:image/png;base64,{image_base64})\n\n**Generated with prompt:** {prompt}"
            else:
                return "Error: No image was generated. Please try again with a different prompt."
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Stable Diffusion. The image generation service may be starting up - please wait a moment and try again."
        except requests.exceptions.Timeout:
            return "Error: Image generation timed out. The server may be busy. Please try again."
        except requests.exceptions.RequestException as e:
            return f"Error generating image: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def generate_image_advanced(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted, ugly, deformed, low resolution",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        sampler_name: str = "Euler a",
    ) -> str:
        """
        Generate an AI image with advanced customization options. Use this when the user wants specific control over image dimensions or generation parameters.
        
        USE THIS TOOL WHEN THE USER SPECIFIES:
        - Custom image dimensions ("make it 768x512", "wide format", "portrait orientation")
        - Quality/speed tradeoff ("higher quality", "quick draft")
        - Specific generation settings
        
        PARAMETERS EXPLAINED:
        - width/height: Image dimensions in pixels. Must be multiples of 64. Common sizes: 512x512 (square), 768x512 (landscape), 512x768 (portrait)
        - steps: More steps = higher quality but slower. 20 is balanced, 30+ for high quality, 10-15 for quick drafts
        - cfg_scale: How closely to follow the prompt. 7 is balanced. Higher (10-15) = more literal, Lower (4-6) = more creative
        - sampler_name: Algorithm for generation. "Euler a" is fast and good. "DPM++ 2M Karras" for higher quality.
        
        :param prompt: Detailed text description of the image. Be specific about subject, style, lighting, mood.
        :param negative_prompt: Things to avoid in the generated image.
        :param width: Image width in pixels (default 512). Must be multiple of 64.
        :param height: Image height in pixels (default 512). Must be multiple of 64.
        :param steps: Sampling steps (default 20). More = better quality but slower. Range: 10-50.
        :param cfg_scale: Prompt guidance strength (default 7.0). Higher = follows prompt more strictly. Range: 1-20.
        :param sampler_name: Sampling method. Options: "Euler a", "DPM++ 2M Karras", "DPM++ SDE Karras", "DDIM".
        :return: The generated image displayed in the chat.
        """
        
        # Ensure dimensions are multiples of 64
        width = max(64, (width // 64) * 64)
        height = max(64, (height // 64) * 64)
        steps = max(1, min(steps, 100))
        cfg_scale = max(1.0, min(cfg_scale, 30.0))
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "seed": -1,
            "sampler_name": sampler_name,
            "batch_size": 1,
            "n_iter": 1,
        }
        
        try:
            api_url = f"{self.valves.STABLE_DIFFUSION_URL}/sdapi/v1/txt2img"
            response = requests.post(api_url, json=payload, timeout=600)
            response.raise_for_status()
            
            result = response.json()
            
            if "images" in result and len(result["images"]) > 0:
                image_base64 = result["images"][0]
                return f"![Generated Image](data:image/png;base64,{image_base64})\n\n**Settings:** {width}x{height}, {steps} steps, CFG {cfg_scale}\n**Prompt:** {prompt}"
            else:
                return "Error: No image was generated."
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Stable Diffusion. Please try again in a moment."
        except requests.exceptions.Timeout:
            return "Error: Image generation timed out. Try reducing steps or image size."
        except requests.exceptions.RequestException as e:
            return f"Error generating image: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
