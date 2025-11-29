import math # funciones matemáticas
import pygame # juegos Pygame (solo usada por el proceso 0)
import numpy as np # para manejo de matrices y cálculos numéricos rápidos
from numba import njit # permite acelerar funciones con compilación JIT
from concurrent.futures import ThreadPoolExecutor # pool de threads para el update de boids en paralelo en cada proceso MPI
from mpi4py import MPI # mpi4py permite comunicación y paralelismo distribuido con MPI
import signal # Manejo de señales del sistema
import sys # Acceso a utilidades del sistema
import time # Para medir tiempos por frame

# ----------------- Settings -----------------
N = 350 # Número total de boids (zombies)
WIDTH, HEIGHT = 900, 700 # Dimensiones de la ventana del juego
VIEW_RADIUS = 40.0 # Radio de visión para reglas de boids
CHASE_RADIUS = 180.0 # Distancia a la que los zombies detectan al jugador
MAX_SPEED = 1.5 # Velocidad máxima de un boid
DT = 0.9 # controla la magnitud del movimiento

# Fuerzas de las reglas de comportamiento de boids
COHESION = 0.003
ALIGNMENT = 0.05
SEPARATION = 0.08
SEP_DIST = 15 # Distancia mínima para separación

PLAYER_SPEED = 5.0 # Velocidad del jugador

LOCAL_THREADS = 4 # Número de threads para paralelizar dentro de cada proceso MPI

# tags para mensajes MPI usados al reiniciar el juego
TAG_POS_RESET = 20
TAG_VEL_RESET = 21

# ----------------- Numba update for a local slice -----------------
# Función acelerada por numba que actualiza un subconjunto local de boids
@njit
def update_local_chunk(pos_all, pos_local, vel_local, player_x, player_y,
                       start, N, VIEW_RADIUS, CHASE_RADIUS, MAX_SPEED, DT,
                       COHESION, ALIGNMENT, SEPARATION, SEP_DIST):

    M = pos_local.shape[0] # Número de boids locales que maneja este proceso

    for local_i in range(M): # Se actualiza cada boid localmente
        i = start + local_i # Índice global del boid dentro del total N

        # Variables acumuladoras para las reglas boid
        center0 = 0.0
        center1 = 0.0
        avoid0 = 0.0
        avoid1 = 0.0
        match0 = 0.0
        match1 = 0.0
        count = 0.0

        # Distancia al jugador
        diff_player_x = player_x - pos_local[local_i, 0]
        diff_player_y = player_y - pos_local[local_i, 1]
        dist_player = (diff_player_x**2 + diff_player_y**2)**0.5

        # Se recorren todos los boids globales (MPI provee pos_all)
        for j in range(N):
            if i == j:
                continue

            diff_x = pos_all[j, 0] - pos_local[local_i, 0]
            diff_y = pos_all[j, 1] - pos_local[local_i, 1]
            dist = (diff_x**2 + diff_y**2)**0.5

            # Boids dentro del radio de visión influyen
            if dist < VIEW_RADIUS:
                center0 += pos_all[j, 0]
                center1 += pos_all[j, 1]

                # Si están muy cerca, se aplica separación
                if dist < SEP_DIST and dist > 1e-8:
                    avoid0 -= diff_x
                    avoid1 -= diff_y

                count += 1.0

        # Si se encontraron vecinos, aplicar cohesión y separación
        if count > 0.0:
            invc = 1.0 / count
            center0 *= invc
            center1 *= invc

            # Cohesión
            vel_local[local_i, 0] += (center0 - pos_local[local_i, 0]) * COHESION
            vel_local[local_i, 1] += (center1 - pos_local[local_i, 1]) * COHESION
            # Separación
            vel_local[local_i, 0] += avoid0 * SEPARATION
            vel_local[local_i, 1] += avoid1 * SEPARATION

        # Si el jugador está cerca, sigue al jugador
        if dist_player < CHASE_RADIUS:
            strength = (CHASE_RADIUS - dist_player) / CHASE_RADIUS
            vel_local[local_i, 0] += diff_player_x * (0.02 * strength)
            vel_local[local_i, 1] += diff_player_y * (0.02 * strength)

        # Limitar velocidad máxima
        speed = (vel_local[local_i, 0]**2 + vel_local[local_i, 1]**2)**0.5
        if speed > MAX_SPEED:
            factor = MAX_SPEED / speed
            vel_local[local_i, 0] *= factor
            vel_local[local_i, 1] *= factor

        # Actualizar posición
        pos_local[local_i, 0] += vel_local[local_i, 0] * DT
        pos_local[local_i, 1] += vel_local[local_i, 1] * DT

        # Warp de pantalla (salen por un lado y entran por el otro)
        if pos_local[local_i, 0] < 0.0: pos_local[local_i, 0] += WIDTH
        if pos_local[local_i, 0] > WIDTH: pos_local[local_i, 0] -= WIDTH
        if pos_local[local_i, 1] < 0.0: pos_local[local_i, 1] += HEIGHT
        if pos_local[local_i, 1] > HEIGHT: pos_local[local_i, 1] -= HEIGHT

# ----------------- Utilities for MPI data exchange -----------------
# Función para dividir equitativamente los N boids entre procesos MPI
def compute_counts_displs(N, size):
    
    base = N // size # Se calcula base = N / size
    rem = N % size # restos para ajustar cuando no divide exacto
    # counts indica cuántos boids le toca a cada proceso
    counts = [base + (1 if r < rem else 0) for r in range(size)]
    # displs indica la posición inicial dentro del arreglo global
    displs = [0] * size
    s = 0
    for r in range(size):
        displs[r] = s
        s += counts[r]

    return counts, displs

# ----------------- Main with MPI -----------------
def main():
    comm = MPI.COMM_WORLD # Se obtiene el comunicador MPI global
    rank = comm.Get_rank() # Número del proceso
    size = comm.Get_size() # Número total de procesos MPI usados

    # Dividir los boids entre procesos
    counts, displs = compute_counts_displs(N, size)
    local_N = counts[rank]
    local_start = displs[rank]
    local_end = local_start + local_N

    # Arreglos locales de posición y velocidad
    pos_local = np.zeros((local_N, 2), dtype=np.float32)
    vel_local = np.zeros((local_N, 2), dtype=np.float32)

    # Solo el rank 0 inicializa posiciones aleatorias y distribuye
    if rank == 0:
        pos_full = np.zeros((N, 2), dtype=np.float32)
        vel_full = np.zeros((N, 2), dtype=np.float32)

        # Posición inicial del jugador
        player_x = WIDTH // 2
        player_y = HEIGHT // 2

        # Inicializar zombies lejos del jugador
        for i in range(N):
            while True:
                p = np.random.rand(2) * np.array([WIDTH, HEIGHT], dtype=np.float32)
                if np.linalg.norm(p - np.array([player_x, player_y], dtype=np.float32)) > CHASE_RADIUS:
                    pos_full[i] = p
                    break

        # Velocidades aleatorias
        vel_full[:] = (np.random.rand(N, 2).astype(np.float32) - 0.5) * 6.0

        # Enviar partes a cada proceso
        for r in range(size):
            s = displs[r]
            e = s + counts[r]
            if r == 0:
                pos_local[:] = pos_full[s:e]
                vel_local[:] = vel_full[s:e]
            else:
                comm.Send([pos_full[s:e].ravel(), MPI.FLOAT], dest=r, tag=10)
                comm.Send([vel_full[s:e].ravel(), MPI.FLOAT], dest=r, tag=11)

    else:
        # Recibir posiciones y velocidades locales en procesos != 0
        if local_N > 0:
            recv_buf = np.empty(local_N * 2, dtype=np.float32)
            comm.Recv(recv_buf, source=0, tag=10)
            pos_local[:] = recv_buf.reshape(local_N, 2)

            recv_buf = np.empty(local_N * 2, dtype=np.float32)
            comm.Recv(recv_buf, source=0, tag=11)
            vel_local[:] = recv_buf.reshape(local_N, 2)

    # Estado del jugador: solo rank 0 tiene control
    if rank == 0:
        player = np.array([WIDTH // 2, HEIGHT // 2], dtype=np.float32)
        player_dir = np.array([0.0, -1.0], dtype=np.float32)
    else:
        player = np.array([0.0, 0.0], dtype=np.float32)
        player_dir = np.array([0.0, -1.0], dtype=np.float32)

    # Pygame solo se inicializa en rank 0
    if rank == 0:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()

        # Cargar imagen de zombie
        enemy_img = pygame.image.load("/Users/maleja/Downloads/proyecto_paralela/zombie.png").convert_alpha()
        enemy_img = pygame.transform.scale(enemy_img, (32, 32))

        # Cargar imagen del jugador
        player_img = pygame.image.load("/Users/maleja/Downloads/proyecto_paralela/human.png").convert_alpha()
        player_img = pygame.transform.scale(player_img, (32, 32))
    else:
        screen = None
        clock = None
        enemy_img = None
        player_img = None

    # Control del bucle de juego
    running = True
    game_over = False

    # Permitir Ctrl + C correctamente
    def sigint_handler(sig, frame):
        nonlocal running
        running = False

    if rank == 0:
        signal.signal(signal.SIGINT, sigint_handler)

    # Preparar buffers de Allgatherv
    recvcounts = [c * 2 for c in counts]
    displs_floats = [d * 2 for d in displs]

    # Crear pool de threads
    executor = ThreadPoolExecutor(max_workers=LOCAL_THREADS)
    try:
        frame = 0
        # Bucle principal MPI + juego
        while running:
            start_time = time.time()
            # Proceso distinto a 0 revisa si rank 0 envió RESET
            if rank != 0:
                if comm.Iprobe(source=0, tag=TAG_POS_RESET):
                    if local_N > 0:
                        recv_buf = np.empty(local_N * 2, dtype=np.float32)
                        comm.Recv(recv_buf, source=0, tag=TAG_POS_RESET)
                        pos_local[:] = recv_buf.reshape(local_N, 2)

                        recv_buf = np.empty(local_N * 2, dtype=np.float32)
                        comm.Recv(recv_buf, source=0, tag=TAG_VEL_RESET)
                        vel_local[:] = recv_buf.reshape(local_N, 2)

            # Rank 0 maneja la entrada del jugador 
            if rank == 0:
                clock.tick(60)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                if not game_over:

                    # Movimiento del jugador
                    keys = pygame.key.get_pressed()
                    move = np.array([0.0, 0.0], dtype=np.float32)

                    if keys[pygame.K_w] or keys[pygame.K_UP]: move[1] -= 1
                    if keys[pygame.K_s] or keys[pygame.K_DOWN]: move[1] += 1
                    if keys[pygame.K_a] or keys[pygame.K_LEFT]: move[0] -= 1
                    if keys[pygame.K_d] or keys[pygame.K_RIGHT]: move[0] += 1

                    # Normalizar y aplicar velocidad
                    if move[0] != 0 or move[1] != 0:
                        norm = np.linalg.norm(move)
                        if norm > 0:
                            mv = move / norm * PLAYER_SPEED
                            player[0] += mv[0]
                            player[1] += mv[1]
                            player_dir[:] = mv / PLAYER_SPEED

                    # Warp del jugador
                    if player[0] < 0: player[0] += WIDTH
                    if player[0] > WIDTH: player[0] -= WIDTH
                    if player[1] < 0: player[1] += HEIGHT
                    if player[1] > HEIGHT: player[1] -= HEIGHT

                # Empaquetar jugador para broadcast
                player_bcast = np.ascontiguousarray(player)

            else:
                # Otros reciben luego el broadcast
                player_bcast = np.empty(2, dtype=np.float32)

            # Enviar jugador a todos
            comm.Bcast(player_bcast, root=0)
            player_x, player_y = float(player_bcast[0]), float(player_bcast[1])

            # Reunir todas las posiciones
            sendbuf = np.ascontiguousarray(pos_local.ravel())
            recvbuf = np.empty(N * 2, dtype=np.float32)
            comm.Allgatherv(sendbuf, [recvbuf, recvcounts, displs_floats, MPI.FLOAT])
            pos_all = recvbuf.reshape(N, 2)

            # Paralelismo -> threads
            if local_N > 0:
                tcount = min(LOCAL_THREADS, local_N)
                base = local_N // tcount
                rem = local_N % tcount
                futures = []
                start_local = 0

                # Dividir boids entre threads
                for t in range(tcount):
                    m = base + (1 if t < rem else 0)
                    if m == 0:
                        continue

                    sub_pos = pos_local[start_local:start_local + m]
                    sub_vel = vel_local[start_local:start_local + m]
                    global_start = local_start + start_local

                    # Ejecutar update_local_chunk en paralelo por thread
                    futures.append(
                        executor.submit(
                            update_local_chunk,
                            pos_all, sub_pos, sub_vel,
                            player_x, player_y, global_start,
                            N, VIEW_RADIUS, CHASE_RADIUS, MAX_SPEED, DT,
                            COHESION, ALIGNMENT, SEPARATION, SEP_DIST
                        )
                    )
                    start_local += m

                # Esperar a que todos los threads terminen
                for f in futures:
                    f.result()

            # Reunir posiciones actualizadas
            sendbuf2 = np.ascontiguousarray(pos_local.ravel())
            recvbuf2 = np.empty(N * 2, dtype=np.float32)
            comm.Allgatherv(sendbuf2, [recvbuf2, recvcounts, displs_floats, MPI.FLOAT])
            pos_all_updated = recvbuf2.reshape(N, 2)

            # Rank 0: dibujar y reset
            if rank == 0:

                if not game_over:
                    # Verificar colisión jugador - zombie
                    diffs = pos_all_updated - np.array([player_x, player_y], dtype=np.float32)
                    dists = np.sqrt(diffs[:,0]**2 + diffs[:,1]**2)
                    if np.any(dists < 10.0):
                        game_over = True

                # Dibujar fondo
                background_image = pygame.image.load('/Users/maleja/Downloads/proyecto_paralela/field.jpg').convert()
                background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
                screen.blit(background_image, (0, 0))

                # Rotar jugador según dirección
                angle_p = math.degrees(math.atan2(player_dir[0], -player_dir[1]))
                rotated_player = pygame.transform.rotate(player_img, angle_p)
                player_rect = rotated_player.get_rect(center=(int(player_x), int(player_y)))
                screen.blit(rotated_player, player_rect)

                # Dibujar todos los boids
                for i in range(N):
                    x, y = pos_all_updated[i]
                    rotated = enemy_img
                    enemy_rect = rotated.get_rect(center=(x, y))
                    screen.blit(rotated, enemy_rect)

                # Pantalla Game Over
                if game_over:
                    font = pygame.font.SysFont('Helvetica', 70)
                    text = font.render("Game Over - Press R", False, 'white')
                    rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
                    screen.blit(text, rect)

                    # Reinicio del juego presionando R
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_r]:

                        # Reiniciar jugador
                        player[0] = WIDTH // 2
                        player[1] = HEIGHT // 2
                        player_dir[:] = np.array([0.0, -1.0], dtype=np.float32)

                        # Generar nuevas posiciones y velocidades
                        pos_full = np.zeros((N, 2), dtype=np.float32)
                        vel_full = np.zeros((N, 2), dtype=np.float32)
                        for i in range(N):
                            while True:
                                p = np.random.rand(2) * np.array([WIDTH, HEIGHT], dtype=np.float32)
                                if np.linalg.norm(p - player) > CHASE_RADIUS:
                                    pos_full[i] = p
                                    break

                        vel_full[:] = (np.random.rand(N, 2).astype(np.float32) - 0.5) * 6.0

                        # Enviar nuevos datos a todos los procesos
                        for r in range(size):
                            s = displs[r]
                            e = s + counts[r]
                            if r == 0:
                                pos_local[:] = pos_full[s:e]
                                vel_local[:] = vel_full[s:e]
                            else:
                                comm.Send([pos_full[s:e].ravel(), MPI.FLOAT], dest=r, tag=TAG_POS_RESET)
                                comm.Send([vel_full[s:e].ravel(), MPI.FLOAT], dest=r, tag=TAG_VEL_RESET)

                        game_over = False

                # Mostrar todo
                pygame.display.flip()

            # Incrementar contador de frames
            frame += 1

    finally:
        # Asegurarse de apagar threads y pygame
        executor.shutdown(wait=True)
        if rank == 0:
            pygame.quit()
        # Apagar MPI
        MPI.Finalize()

# Ejecutar main
if __name__ == "__main__":
    main()