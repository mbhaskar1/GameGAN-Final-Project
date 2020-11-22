import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PIXEL_WIDTH = 40


class PongDisplay:
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((PIXEL_WIDTH * width, PIXEL_WIDTH * height))
        self.screen.fill(BLACK)
        pygame.display.flip()

    def update_screen(self, pixels):
        for i in range(self.width):
            for j in range(self.height):
                pixel = pixels[i, j]
                if pixel == 0:
                    pygame.draw.rect(self.screen, BLACK, (PIXEL_WIDTH * i, PIXEL_WIDTH * j, PIXEL_WIDTH, PIXEL_WIDTH))
                else:
                    pygame.draw.rect(self.screen, WHITE, (PIXEL_WIDTH * i, PIXEL_WIDTH * j, PIXEL_WIDTH, PIXEL_WIDTH))
        pygame.display.update()
