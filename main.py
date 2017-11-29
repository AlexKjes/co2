import model


room = model.Room([3, 3, 4], 500, 500, 4, 6)


for _ in range(8*60*60):
    room.time_step()


print(room.co2ppm)
