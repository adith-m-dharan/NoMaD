from rclpy.node import Node

class ROSData:
    def __init__(self, node: Node, timeout: int = 3, queue_size: int = 1, name: str = ""):
        self.node = node
        self.timeout = timeout
        self.last_time_received = float("-inf")
        self.queue_size = queue_size
        self.data = None
        self.name = name
        self.phantom = False

    def get(self):
        return self.data

    def set(self, data):
        current_time = self.node.get_clock().now().seconds_nanoseconds()[0]
        time_waited = current_time - self.last_time_received
        if self.queue_size == 1:
            self.data = data
        else:
            if self.data is None or time_waited > self.timeout:  # reset queue if timeout
                self.data = []
            if len(self.data) == self.queue_size:
                self.data.pop(0)
            self.data.append(data)
        self.last_time_received = current_time

    def is_valid(self, verbose: bool = False):
        current_time = self.node.get_clock().now().seconds_nanoseconds()[0]
        time_waited = current_time - self.last_time_received
        valid = time_waited < self.timeout
        if self.queue_size > 1:
            valid = valid and len(self.data) == self.queue_size
        if verbose and not valid:
            self.node.get_logger().warn(f"Not receiving {self.name} data for {time_waited} seconds (timeout: {self.timeout} seconds)")
        return valid
