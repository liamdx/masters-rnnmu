for k, v in processedMidi.items():
    # Print an empirical estimate of its global tempo
    print("Current Song: " + k)
    print("Tempo = %f" % v.estimate_tempo())
    # Compute the relative amount of each semitone across the entire song, a proxy for key
    total_velocity = sum(sum(v.get_chroma()))
    print([sum(semitone)/total_velocity for semitone in v.get_chroma()])