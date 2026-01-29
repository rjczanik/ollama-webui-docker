"""
title: Stable Diffusion Image Generator
author: rjczanik
version: 1.2.0
license: MIT
description: Generate AI images from text descriptions using Stable Diffusion.
requirements: requests
"""

import requests
from pydantic import BaseModel, Field


class Valves(BaseModel):
    STABLE_DIFFUSION_URL: str = Field(
        default="http://stable-diffusion:17860",
        description="The URL of the Stable Diffusion WebUI API"
    )


class Tools:
    def __init__(self):
        self.valves = Valves()

    def generate_image(self, prompt: str) -> str:
        """
        Generate an AI image from a text description using Stable Diffusion.
        Use this tool when the user asks to create, generate, draw, or make an image, picture, artwork, or illustration.
        
        :param prompt: Detailed text description of the image to generate.
        :return: The generated image or an error message.
        """
        payload = {
            "prompt": prompt,
            "negative_prompt": "blurry, bad quality, distorted, ugly, deformed",
            "steps": 20,
            "cfg_scale": 7.0,
            "width": 512,
            "height": 512,
            "seed": -1,
            "sampler_name": "Euler a",
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
                return f"![Generated Image](data:image/png;base64,{image_base64})"
            else:
                return "Error: No image was generated."
                
        except Exception as e:
            return f"Error: {str(e)}"
