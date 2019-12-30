import click
import os
from presto_chango.database import create_database
from presto_chango.app import identify_song


@click.group(help='Music identification through audio fingerprinting')
def cli():
    pass


@cli.command()
@click.argument('songs-dir', type=click.Path(exists=True), required=True)
def create_db(songs_dir):
    create_database(songs_dir)


@cli.command()
@click.option('--file', type=click.Path(exists=True), help="Location of pre-recorded sample")
def identify(file):
    identify_song(file)
