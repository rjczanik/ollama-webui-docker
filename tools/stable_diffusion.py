"""
title: Stable Diffusion Image Generator
author: rjczanik
version: 1.0.0
description: Generate images using Stable Diffusion via AUTOMATIC1111 WebUI API
"""

import os
import base64
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
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: int = -1,
        sampler_name: Optional[str] = None,
    ) -> str:
        """
        Generate an image using Stable Diffusion based on the given prompt.
        
        :param prompt: The text description of the image to generate. Be detailed and descriptive.
        :param negative_prompt: Things to avoid in the generated image (e.g., "blurry, low quality, distorted").
        :param width: Image width in pixels (default: 512). Should be a multiple of 64.
        :param height: Image height in pixels (default: 512). Should be a multiple of 64.
        :param steps: Number of sampling steps (default: 20). More steps = better quality but slower.
        :param cfg_scale: Classifier-free guidance scale (default: 7.0). Higher = more prompt adherence.
        :param seed: Random seed for reproducibility (-1 for random).
        :param sampler_name: Sampling method (default: "Euler a").
        :return: A message with the generated image in markdown format or an error message.
        """
        
        # Use defaults from valves if not specified
        width = width or self.valves.DEFAULT_WIDTH
        height = height or self.valves.DEFAULT_HEIGHT
        steps = steps or self.valves.DEFAULT_STEPS
        cfg_scale = cfg_scale or self.valves.DEFAULT_CFG_SCALE
        sampler_name = sampler_name or self.valves.DEFAULT_SAMPLER
        
        # Ensure dimensions are multiples of 64
        width = (width // 64) * 64
        height = (height // 64) * 64
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "blurry, bad quality, distorted, ugly, deformed",
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "sampler_name": sampler_name,
            "batch_size": 1,
            "n_iter": 1,
        }
        
        try:
            api_url = f"{self.valves.STABLE_DIFFUSION_URL}/sdapi/v1/txt2img"
            response = requests.post(api_url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            
            if "images" in result and len(result["images"]) > 0:
                image_base64 = result["images"][0]
                # Return markdown image with base64 data
                return f"![Generated Image](data:image/png;base64,{image_base64})\n\n**Prompt:** {prompt}\n**Seed:** {result.get('info', {}).get('seed', 'N/A') if isinstance(result.get('info'), dict) else seed}"
            else:
                return "Error: No image was generated. The API response did not contain any images."
                
        except requests.exceptions.ConnectionError:
            return f"Error: Could not connect to Stable Diffusion API at {self.valves.STABLE_DIFFUSION_URL}. Please ensure the service is running."
        except requests.exceptions.Timeout:
            return "Error: Request timed out. Image generation is taking too long."
        except requests.exceptions.RequestException as e:
            return f"Error generating image: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def get_sd_models(self) -> str:
        """
        Get the list of available Stable Diffusion models/checkpoints.
        
        :return: A list of available models or an error message.
        """
        try:
            api_url = f"{self.valves.STABLE_DIFFUSION_URL}/sdapi/v1/sd-models"
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            models = response.json()
            
            if models:
                model_list = "\n".join([f"- {m.get('model_name', m.get('title', 'Unknown'))}" for m in models])
                return f"**Available Stable Diffusion Models:**\n{model_list}"
            else:
                return "No models found. Please add models to the Stable Diffusion models directory."
                
        except requests.exceptions.RequestException as e:
            return f"Error fetching models: {str(e)}"

    def get_sd_samplers(self) -> str:
        """
        Get the list of available sampling methods.
        
        :return: A list of available samplers or an error message.
        """
        try:
            api_url = f"{self.valves.STABLE_DIFFUSION_URL}/sdapi/v1/samplers"
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            samplers = response.json()
            
            if samplers:
                sampler_list = "\n".join([f"- {s.get('name', 'Unknown')}" for s in samplers])
                return f"**Available Samplers:**\n{sampler_list}"
            else:
                return "No samplers found."
                
        except requests.exceptions.RequestException as e:
            return f"Error fetching samplers: {str(e)}"
