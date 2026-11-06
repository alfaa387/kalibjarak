#!/usr/bin/env python3

import os
import struct
import time
from collections import deque
from datetime import datetime

import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from robotis_controller_msgs.msg import WriteControlTable
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Int32, String

DEFAULT_PARAM_FILE = "/home/aroc/kalibjarakgavragayuhuy/param.yaml"
DEFAULT_TARGET_TO_KICK = 30.0
TARGET_MIN_CM = 5.0
TARGET_MAX_CM = 60.0
TARGET_DISTANCE = 43.0
PAN_THRESHOLD = 0.1

INIT_POSE_PAGE = 9
BRINGUP_POSE_PAGE = 15
STAND_PAGE = 80
KICK_PAGE_KIRI = 155
KICK_PAGE_KANAN = 156

ACTION_DURATION = 1.0
BUTTON_DEBOUNCE = 0.4
FILTER_WINDOW = 5
FILTER_ALPHA = 0.2
BALL_VOTE_WINDOW = 5
BALL_MIN_VOTES = 3

RGB_COLORS = {
    'OFF': (0, 0, 0),
    'RED': (31, 0, 0),
    'GREEN': (0, 31, 0),
    'BLUE': (0, 0, 31),
    'YELLOW': (31, 31, 0),
    'CYAN': (0, 31, 31),
}


class KalibrasiJarakNode(Node):
    def __init__(self):
        super().__init__('kalibrasi_jarak_node')

        self.declare_parameter('param_file', DEFAULT_PARAM_FILE)
        self.param_file = os.path.expanduser(
            self.get_parameter(
                'param_file').get_parameter_value().string_value)

        self.module_pub = self.create_publisher(
            String, '/robotis/enable_ctrl_module', 10)
        self.action_pub = self.create_publisher(
            Int32, '/robotis/action/page_num', 10)
        self.walking_command_pub = self.create_publisher(
            String, '/robotis/walking/command', 10)
        self.head_command_pub = self.create_publisher(
            String, '/communication/head_command', 10)
        self.state_pub = self.create_publisher(
            String, '/communication/robot_state', 10)
        self.led_pub = self.create_publisher(
            WriteControlTable, '/robotis/write_control_table', 10)

        self.create_subscription(
            String, '/robotis/open_cr/button', self.button_callback, 10)
        self.create_subscription(
            Float32, '/vision/hasil_servo', self.jarak_callback, 10)
        self.create_subscription(
            JointState, '/robotis/goal_joint_states', self.joint_callback, 10)
        self.create_subscription(
            String, '/vision/bola_posisi', self.ball_position_callback, 10)

        self.jarak_bola = 0.0
        self.head_pan = 0.0
        self.ball_posisi = ""
        self.ball_posisi_stable = ""

        self.dist_buffer = deque(maxlen=FILTER_WINDOW)
        self.dist_ema = None

        self.ball_position_history = deque(maxlen=BALL_VOTE_WINDOW)

        self.target_to_kick = DEFAULT_TARGET_TO_KICK
        self.captured_distance_cm = DEFAULT_TARGET_TO_KICK
        self.updated_at = None
        self.load_param()

        self.is_kicking = False
        self.is_standing = False
        self.last_press_time = 0.0

        self.led1 = False
        self.led2 = False
        self.led3 = False
        self.rgb_r = 0
        self.rgb_g = 0
        self.rgb_b = 0

        self.create_timer(2.0, self.status_log)

        self.publish_state("NORMAL")
        self.set_status_led(False, False, False)
        self.set_rgb_color('OFF')

        self.get_logger().info("=" * 68)
        self.get_logger().info("KALIBRASI TARGET_TO_KICK - kontrol tombol:")
        self.get_logger().info(
            "  mode  : capture jarak bola sekarang -> simpan param.yaml")
        self.get_logger().info(
            "  start : test kick langsung di tempat (tanpa jalan)")
        self.get_logger().info(
            "  user  : init pose (page 9) / bringup pose (page 15)")
        self.get_logger().info(f"  param : {self.param_file}")
        self.get_logger().info(
            f"  target aktif: {self.target_to_kick:.2f} cm")
        if self.updated_at:
            self.get_logger().info(f"  terakhir diubah: {self.updated_at}")
        self.get_logger().info(
            "Jalankan `ros2 run GoNasional headcontrol` di terminal lain")
        self.get_logger().info("=" * 68)

    # ==========================================================
    # PARAM YAML
    # ==========================================================
    def load_param(self):
        if not os.path.isfile(self.param_file):
            self.get_logger().warn(
                f"param.yaml belum ada ({self.param_file}), "
                f"pakai default {DEFAULT_TARGET_TO_KICK:.1f} cm"
            )
            return

        try:
            with open(self.param_file, 'r') as file:
                data = yaml.safe_load(file) or {}

            target = float(data.get('target_to_kick', DEFAULT_TARGET_TO_KICK))
            if not (TARGET_MIN_CM <= target <= TARGET_MAX_CM):
                self.get_logger().warn(
                    f"target_to_kick di param.yaml di luar rentang "
                    f"{TARGET_MIN_CM:.0f}-{TARGET_MAX_CM:.0f} cm "
                    f"({target:.2f}), pakai default"
                )
                target = DEFAULT_TARGET_TO_KICK

            self.target_to_kick = target

            captured = data.get('captured_distance_cm')
            self.captured_distance_cm = (
                float(captured) if captured is not None else target)
            self.updated_at = data.get('updated_at')

            self.get_logger().info(
                f"Param dimuat: target_to_kick={self.target_to_kick:.2f} cm"
            )
        except Exception as exc:
            self.get_logger().error(
                f"Gagal membaca param.yaml: {exc}. "
                f"Pakai default {DEFAULT_TARGET_TO_KICK:.1f} cm"
            )
            self.target_to_kick = DEFAULT_TARGET_TO_KICK
            self.captured_distance_cm = DEFAULT_TARGET_TO_KICK
            self.updated_at = None

    def save_param(self, captured_distance):
        data = {
            'target_to_kick': round(float(self.target_to_kick), 2),
            'captured_distance_cm': round(float(captured_distance), 2),
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }

        try:
            param_path = os.path.abspath(self.param_file)
            os.makedirs(os.path.dirname(param_path), exist_ok=True)
            temp_path = param_path + '.tmp'
            with open(temp_path, 'w') as file:
                yaml.safe_dump(data, file, sort_keys=False)
            os.replace(temp_path, param_path)

            self.captured_distance_cm = data['captured_distance_cm']
            self.updated_at = data['updated_at']
            return True
        except Exception as exc:
            self.get_logger().error(
                f"Gagal menyimpan param.yaml: {exc}")
            return False

    # ==========================================================
    # CALLBACK SENSOR
    # ==========================================================
    def jarak_callback(self, msg):
        value = float(msg.data)

        if value <= 0.0:
            self.dist_buffer.clear()
            self.dist_ema = None
            self.jarak_bola = 0.0
            return

        self.dist_buffer.append(value)
        median = sorted(self.dist_buffer)[len(self.dist_buffer) // 2]

        if self.dist_ema is None:
            self.dist_ema = median
        else:
            self.dist_ema = (
                FILTER_ALPHA * median
                + (1.0 - FILTER_ALPHA) * self.dist_ema)

        self.jarak_bola = self.dist_ema

    def joint_callback(self, msg):
        for name, position in zip(msg.name, msg.position):
            if name == 'head_pan':
                self.head_pan = position

    def ball_position_callback(self, msg):
        posisi = msg.data.lower().strip()

        if posisi not in ('kiri', 'tengah', 'kanan'):
            self.get_logger().warn(
                f"Posisi bola tidak dikenal: {msg.data}")
            return

        self.ball_position_history.append(posisi)

        kiri = self.ball_position_history.count('kiri')
        tengah = self.ball_position_history.count('tengah')
        kanan = self.ball_position_history.count('kanan')
        votes = {'kiri': kiri, 'tengah': tengah, 'kanan': kanan}
        posisi_terbanyak = max(votes, key=votes.get)

        if votes[posisi_terbanyak] >= BALL_MIN_VOTES:
            self.ball_posisi_stable = posisi_terbanyak

        self.ball_posisi = posisi

    # ==========================================================
    # TOMBOL
    # ==========================================================
    def button_callback(self, msg):
        button = msg.data.strip().lower()
        now = time.time()

        if now - self.last_press_time < BUTTON_DEBOUNCE:
            return

        self.last_press_time = now

        if button == 'mode':
            self.capture_target()
        elif button == 'start':
            self.test_kick()
        elif button == 'user':
            self.toggle_pose()
        else:
            self.get_logger().warn(f"Tombol tidak dikenal: {msg.data}")

    def capture_target(self):
        if self.is_kicking:
            self.get_logger().warn(
                "Capture diabaikan: robot sedang test kick")
            return

        jarak = self.jarak_bola
        self.get_logger().info(
            f"CAPTURE | jarak bola terukur = {jarak:.2f} cm")

        if jarak <= 0.0:
            self.get_logger().warn(
                "Capture gagal: bola belum terdeteksi / belum ada data "
                "/vision/hasil_servo"
            )
            self.set_rgb_color('RED')
            return

        if not (TARGET_MIN_CM <= jarak <= TARGET_MAX_CM):
            self.get_logger().warn(
                f"Capture gagal: jarak {jarak:.2f} cm di luar rentang "
                f"{TARGET_MIN_CM:.0f}-{TARGET_MAX_CM:.0f} cm"
            )
            self.set_rgb_color('RED')
            return

        if jarak >= TARGET_DISTANCE:
            self.get_logger().warn(
                f"Catatan: jarak {jarak:.2f} cm >= TARGET_DISTANCE "
                f"gonas ({TARGET_DISTANCE:.0f} cm)"
            )

        old_target = self.target_to_kick
        self.target_to_kick = round(jarak, 2)

        if self.save_param(self.target_to_kick):
            self.get_logger().info(
                f"SAVED | TARGET_TO_KICK {old_target:.2f} -> "
                f"{self.target_to_kick:.2f} cm | file: {self.param_file}"
            )
            self.set_rgb_color('GREEN')
        else:
            self.target_to_kick = old_target
            self.set_rgb_color('RED')

    def test_kick(self):
        if self.is_kicking:
            self.get_logger().warn(
                "Test kick diabaikan: masih ada proses kick")
            return

        self.is_kicking = True
        try:
            kick_page = self.decide_kick_side()
            kick_name = (
                'KIRI' if kick_page == KICK_PAGE_KIRI else 'KANAN')

            self.get_logger().info(
                f"TEST KICK | target_yaml={self.target_to_kick:.2f} cm | "
                f"jarak_sekarang={self.jarak_bola:.2f} cm | "
                f"sisi={kick_name}"
            )

            self.set_rgb_color('BLUE')
            self.send_walking_command("stop")
            self.publish_head_command("CENTER")
            time.sleep(0.5)

            self.init_pose(STAND_PAGE)
            time.sleep(1.0)

            self.publish_state("RECOVER")
            time.sleep(0.5)

            self.publish_head_command("CENTER")
            time.sleep(0.5)

            self.enable_action_module()
            time.sleep(0.5)

            self.init_pose(kick_page)
            self.get_logger().info(
                f"KICK {kick_name} dijalankan (page {kick_page})")
            time.sleep(ACTION_DURATION)
        finally:
            self.enable_walking_module()
            self.publish_state("NORMAL")
            self.publish_head_command("UNLOCK")
            self.init_pose(STAND_PAGE)
            time.sleep(0.5)
            self.set_rgb_color('OFF')
            self.is_kicking = False
            self.get_logger().info(
                "Test kick selesai, robot kembali berdiri")

    def toggle_pose(self):
        if self.is_kicking:
            self.get_logger().warn(
                "Init pose diabaikan: robot sedang test kick")
            return

        self.enable_action_module()
        time.sleep(0.5)

        if not self.is_standing:
            self.init_pose(INIT_POSE_PAGE)
            self.is_standing = True
            self.set_status_led(led3=True)
            self.get_logger().info(
                f"USER -> INIT POSE (page {INIT_POSE_PAGE})")
        else:
            self.init_pose(BRINGUP_POSE_PAGE)
            self.is_standing = False
            self.set_status_led(led3=False)
            self.get_logger().info(
                f"USER -> BRINGUP POSE (page {BRINGUP_POSE_PAGE})")

    def decide_kick_side(self):
        posisi = self.ball_posisi_stable
        pan = self.head_pan

        if posisi == 'kiri':
            self.get_logger().info(
                f"DECISION CAMERA -> KIRI (pan={pan:.2f})")
            return KICK_PAGE_KIRI

        if posisi == 'kanan':
            self.get_logger().info(
                f"DECISION CAMERA -> KANAN (pan={pan:.2f})")
            return KICK_PAGE_KANAN

        if posisi == 'tengah':
            if pan > PAN_THRESHOLD:
                self.get_logger().info(
                    f"CAMERA TENGAH + PAN {pan:.2f} -> KIRI")
                return KICK_PAGE_KIRI
            self.get_logger().info(
                f"CAMERA TENGAH + PAN {pan:.2f} -> KANAN")
            return KICK_PAGE_KANAN

        self.get_logger().warn(
            "ball_posisi_stable belum tersedia, fallback ke head_pan")

        if pan > PAN_THRESHOLD:
            return KICK_PAGE_KIRI
        if pan < -PAN_THRESHOLD:
            return KICK_PAGE_KANAN
        return KICK_PAGE_KANAN

    def status_log(self):
        if self.is_kicking:
            return

        self.get_logger().info(
            f"STATUS | jarak={self.jarak_bola:.2f} cm | "
            f"target={self.target_to_kick:.2f} cm | "
            f"bola={self.ball_posisi_stable or '-'} | "
            f"pan={self.head_pan:.2f}"
        )

    # ==========================================================
    # PUBLISHER HELPER
    # ==========================================================
    def enable_action_module(self):
        self.module_pub.publish(String(data='action_module'))

    def enable_walking_module(self):
        self.module_pub.publish(String(data='walking_module'))

    def send_walking_command(self, command):
        self.walking_command_pub.publish(String(data=command))

    def init_pose(self, page):
        self.action_pub.publish(Int32(data=int(page)))

    def publish_head_command(self, command):
        self.head_command_pub.publish(String(data=command))

    def publish_state(self, state):
        self.state_pub.publish(String(data=state))

    # ==========================================================
    # LED
    # ==========================================================
    def set_status_led(self, led1=None, led2=None, led3=None):
        if led1 is not None:
            self.led1 = bool(led1)
        if led2 is not None:
            self.led2 = bool(led2)
        if led3 is not None:
            self.led3 = bool(led3)

        value = (
            (int(self.led1) << 0)
            | (int(self.led2) << 1)
            | (int(self.led3) << 2)
        )

        msg = WriteControlTable()
        msg.joint_name = 'open-cr'
        msg.start_item_name = 'LED'
        msg.data_length = 1
        msg.data = [value & 0xFF]
        self.led_pub.publish(msg)

    def set_rgb_led(self, r=None, g=None, b=None):
        if r is not None:
            self.rgb_r = r & 0x1F
        if g is not None:
            self.rgb_g = g & 0x1F
        if b is not None:
            self.rgb_b = b & 0x1F

        rgb_value = (
            (self.rgb_r << 10)
            | (self.rgb_g << 5)
            | self.rgb_b
        )

        msg = WriteControlTable()
        msg.joint_name = 'open-cr'
        msg.start_item_name = 'LED_RGB'
        msg.data_length = 2
        msg.data = list(struct.pack('<H', rgb_value))
        self.led_pub.publish(msg)

    def set_rgb_color(self, color):
        color = color.upper()

        if color not in RGB_COLORS:
            self.get_logger().warn(
                f"RGB color '{color}' tidak ditemukan")
            return

        r, g, b = RGB_COLORS[color]
        self.set_rgb_led(r, g, b)


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = KalibrasiJarakNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.set_rgb_color('OFF')
                node.set_status_led(False, False, False)
            except Exception:
                pass
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
