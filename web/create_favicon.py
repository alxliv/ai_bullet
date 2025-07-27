#!/usr/bin/env python3
"""
Simple script to create a chat favicon without external dependencies
"""

# Create a simple SVG that we can convert to ICO later
svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
  <!-- Chat bubble -->
  <ellipse cx="16" cy="14" rx="12" ry="8" fill="#0084ff" stroke="#0066cc" stroke-width="1"/>
  <!-- Tail -->
  <polygon points="8,20 12,26 16,20" fill="#0084ff"/>
  <!-- Dots for text -->
  <circle cx="11" cy="14" r="1.5" fill="white"/>
  <circle cx="16" cy="14" r="1.5" fill="white"/>
  <circle cx="21" cy="14" r="1.5" fill="white"/>
</svg>'''

with open('favicon.svg', 'w') as f:
    f.write(svg_content)

print("Created favicon.svg - you can convert this to ICO format online at:")
print("- https://convertio.co/svg-ico/")
print("- https://cloudconvert.com/svg-to-ico")
print("- Or use it directly as <link rel='icon' type='image/svg+xml' href='/favicon.svg'>")
