import model


room = model.Room([3, 3, 4], 500, 500, 3, 2)


for _ in range(5*60*60):
    room.time_step()


print(room.co2ppm)
