    for i, instruction in enumerate(instructions):

        tracker = CarbonTrackerManual(epochs=1, monitor_epochs=1, update_interval=1,
            components='all', epochs_before_pred=1, verbose=2)
        tracker.tracker.pue_manual=1
        tracker.intensity_updater.ci_manual = 100

        tracker.epoch_start()
        print(f"Prompt {i}")

        outputs = model.generate(instruction, sampling_params=sampling_param)
        

        tracker.epoch_end()
        [energy, co2] = tracker.get_energy_co2()

        info[str(instruction)] = {
            "Energy": energy,
            "CO2": co2
        }