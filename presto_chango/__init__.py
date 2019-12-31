import click
import os
from presto_chango.database import create_database
from presto_chango.app import identify_song
from appdirs import AppDirs
from presto_chango.tester import test_accuracy


@click.group(help='Music identification through audio fingerprinting')
def cli():
    if os.path.exists(AppDirs('Presto-Chango').user_data_dir):
        pass
    else:
        os.mkdir(AppDirs('Presto-Chango').user_data_dir)


@cli.command()
@click.argument('songs-dir', type=click.Path(exists=True), required=True)
def create_db(songs_dir):
    create_database(songs_dir)


@cli.command()
@click.option('--file', type=click.Path(exists=True), help="Location of pre-recorded sample")
def identify(file):
    identify_song(file)


@cli.command()
def test():
    test_accuracy()
