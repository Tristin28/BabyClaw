import pygame
from pygame.locals import *
import random

# Initialize Pygame
pygame.init()

# Set up display
screen_width = 400
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Flappy Bird")

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)

# Load images
bird_img = pygame.image.load("bird.png")
pipe_img = pygame.image.load("pipe.png")

# Set up game variables
clock = pygame.time.Clock()
bird_x = 100
bird_y = 300
bird_velocity = 5
gravity = 1

pipes = []
pipe_width = 70
pipe_height = 400
pipe_gap = 200
pipe_frequency = 1500  # milliseconds

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        if event.type == KEYDOWN and event.key == K_SPACE:
            bird_velocity = -10

    # Update bird position
    bird_y += bird_velocity
    bird_velocity += gravity

    # Generate pipes
    time_since_last_pipe = pygame.time.get_ticks() - pipe_frequency
    if time_since_last_pipe > 0:
        pipe_x = screen_width
        pipe_y = random.randint(150, 450)
        pipes.append((pipe_x, pipe_y))
        pipe_frequency = pygame.time.get_ticks()

    # Move pipes
    for i, (pipe_x, pipe_y) in enumerate(pipes):
        pipe_x -= 5
        if pipe_x < -pipe_width:
            pipes.pop(i)

    # Draw everything
    screen.fill(white)
    screen.blit(bird_img, (bird_x, bird_y))
    for pipe_x, pipe_y in pipes:
        screen.blit(pipe_img, (pipe_x, pipe_y))
        screen.blit(pygame.transform.flip(pipe_img, False, True), (pipe_x, pipe_y + pipe_gap))

    pygame.display.update()
    clock.tick(30)

pygame.quit()