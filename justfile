default: list

# List available commands
list:
    @just --list --unsorted

bench-all:
    @cd bench && make run

bench-micro:
    @cd bench && make run-micro

bench-composite:
    @cd bench && make run-composite

bench-mnist:
    @cd bench && make run-mnist
