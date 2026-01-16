using UnityEngine;


namespace GBGST.Scripts
{
    public class FreeCameraMovement : MonoBehaviour
    {
        public float moveSpeed = 5f;
        public float lookSpeed = 2f;

        private float yaw = 0f;
        private float pitch = 0f;

        void Update()
        {
            float horizontal = Input.GetAxis("Horizontal");
            float vertical = Input.GetAxis("Vertical");

            // Movement
            Vector3 move = transform.right * horizontal + transform.forward * vertical;
            if (Input.GetKey(KeyCode.E)) move += Vector3.up;
            if (Input.GetKey(KeyCode.Q)) move -= Vector3.up;

            transform.position += move * (moveSpeed * Time.deltaTime);

            // Rotation
            yaw += lookSpeed * Input.GetAxis("Mouse X");
            pitch -= lookSpeed * Input.GetAxis("Mouse Y");
            pitch = Mathf.Clamp(pitch, -90f, 90f);

            transform.eulerAngles = new Vector3(pitch, yaw, 0f);
        }
    }
}