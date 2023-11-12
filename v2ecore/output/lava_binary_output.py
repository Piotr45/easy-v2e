import numpy as np
import logging

from engineering_notation import EngNumber  # only from pip

# check https://gitlab.com/inivation/dv/dv-processing to install dv-processing-python
import dv_processing as dv

from v2ecore.v2e_utils import v2e_quit

logger = logging.getLogger(__name__)


class LAVABinaryOutput:
    """
    Outputs LavaBinaryOutput format DVS data from v2e
    """

    def __init__(self, filepath: str, output_width=640, output_height=480):
        self.filepath = filepath
        self.numEventsWritten = 0
        self.numOnEvents = 0
        self.numOffEvents = 0
        logging.info(
            "opening LAVA binary output file {} in binary mode".format(filepath)
        )

        self.flipy = False
        self.flipx = False
        self.sizex = output_width
        self.sizey = output_height

        self.store = dv.EventStore()

    def cleanup(self):
        self.close()

    def close(self):
        self.encode_2d_spikes(self.filepath, self.store.numpy())

    def appendEvents(self, events: np.ndarray, signnoise_label: np.ndarray = None):
        """Append events to LAVA binary output

        Parameters
        ----------
        events: np.ndarray if any events, else None
            [N, 4], each row contains [timestamp, x coordinate, y coordinate, sign of event (+1 ON, -1 OFF)].
            NOTE x,y, NOT y,x.
        signnoise: np.ndarray
          [N] each entry is 1 for signal or 0 for noise

        Returns
        -------
        None
        """

        if len(events) == 0:
            return
        n = events.shape[0]
        for event in events:
            t = int(event[0] * 1e6)
            x = int(event[1])
            if self.flipx:
                x = (self.sizex - 1) - x  # 0 goes to sizex-1
            y = int(event[2])
            if self.flipy:
                y = (self.sizey - 1) - y
            p = int((event[3] + 1) / 2)  # 0=off, 1=on

            try:
                self.store.push_back(t, x, y, p)
            except RuntimeError as e:
                logger.warning("caught exception event {} to store".format(e))

            if p == 1:
                self.numOnEvents += 1
            else:
                self.numOffEvents += 1
            self.numEventsWritten += 1

        # logger.info('wrote {} events'.format(n))

    def encode_2d_spikes(self, filename, td_event):
        """Writes two dimensional binary spike file from a td_event event.
        It is the same format used in neuromorphic datasets NMNIST & NCALTECH101.

        The binary file is encoded as follows:
            * Each spike event is represented by a 40 bit number.
            * First 8 bits (bits 39-32) represent the xID of the neuron.
            * Next 8 bits (bits 31-24) represent the yID of the neuron.
            * Bit 23 represents the sign of spike event: 0=>OFF event, 1=>ON event.
            * The last 23 bits (bits 22-0) represent the spike event timestamp in
            microseconds.

        Parameters
        ----------
        filename : str
            name of spike file.
        td_event : event
            spike event object

        Examples
        --------

        >>> encode_2d_spikes(file_path, td_event)
        """
        events = np.array([list(td) for td in td_event])
        t, x, y, c = (
            np.array(events[:, 0]),
            np.array(events[:, 1]),
            np.array(events[:, 2]),
            np.array(events[:, 3]),
        )
        x_event = np.round(x).astype(int)
        y_event = np.round(y).astype(int)
        c_event = np.round(c).astype(int)
        # encode spike time in us
        t_event = np.round(t * 1000).astype(int)
        output_byte_array = bytearray(len(t_event) * 5)
        output_byte_array[0::5] = np.uint8(x_event).tobytes()
        output_byte_array[1::5] = np.uint8(y_event).tobytes()
        output_byte_array[2::5] = np.uint8(
            ((t_event >> 16) & 0x7F) | (c_event.astype(int) << 7)
        ).tobytes()
        output_byte_array[3::5] = np.uint8((t_event >> 8) & 0xFF).tobytes()
        output_byte_array[4::5] = np.uint8(t_event & 0xFF).tobytes()
        with open(filename, "wb") as output_file:
            output_file.write(output_byte_array)


if __name__ == "__main__":

    class LAVABinaryOutputTt:
        f = LAVABinaryOutput("lava_spikes.bin")
        e = [
            [1, 400, 0, 0],
            [2, 0, 400, 0],
            [3, 300, 400, 0],
            [4, 400, 300, 1],
            [5, 400, 300, 0],
        ]
        ne = np.array(e)
        eventsNum = 2000 * 5
        nne = np.tile(ne, (int(eventsNum / 5), 1))
        nne[:, 0] = np.arange(1, eventsNum + 1)
        f.appendEvents(nne)
        print("wrote {} events".format(nne.shape[0]))
        f.close()
