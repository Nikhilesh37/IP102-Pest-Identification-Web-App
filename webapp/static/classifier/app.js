const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("image-input");

if (dropzone && fileInput) {
    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            event.stopPropagation();
            dropzone.classList.add("drag-active");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            event.stopPropagation();
            dropzone.classList.remove("drag-active");
        });
    });

    dropzone.addEventListener("drop", (event) => {
        if (!event.dataTransfer || !event.dataTransfer.files.length) {
            return;
        }
        fileInput.files = event.dataTransfer.files;
    });
}
