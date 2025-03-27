from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from payos import PayOS, ItemData, PaymentData
import time
import os

# Lấy thông tin cấu hình từ biến môi trường
PAYOS_CLIENT_ID = os.getenv("PAYOS_CLIENT_ID", "")
PAYOS_API_KEY = os.getenv("PAYOS_API_KEY", "")
PAYOS_CHECKSUM_KEY = os.getenv("PAYOS_CHECKSUM_KEY", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "")

# Khởi tạo đối tượng PayOS
payos = PayOS(client_id=PAYOS_CLIENT_ID, api_key=PAYOS_API_KEY, checksum_key=PAYOS_CHECKSUM_KEY)

# Khởi tạo APIRouter
router = APIRouter(prefix="/payments", tags=["payments"])

# Định nghĩa model cho input của /create-payment
class PaymentRequest(BaseModel):
    amount: int  # Số tiền cần thanh toán (VND)
    description: str  # Mô tả đơn hàng

# Định nghĩa model cho output của /create-payment
class PaymentResponse(BaseModel):
    success: bool
    checkout_url: str
    order_code: int

# Định nghĩa model cho input của /check-payment
class PaymentStatusRequest(BaseModel):
    order_code: int  # Mã đơn hàng để kiểm tra

# Định nghĩa model cho output của /check-payment
class PaymentStatusResponse(BaseModel):
    success: bool
    message: str

# Endpoint tạo liên kết thanh toán
@router.post("/create-payment", response_model=PaymentResponse)
async def create_payment(payment: PaymentRequest):
    try:
        # Tạo orderCode dựa trên timestamp (đảm bảo hợp lệ)
        order_code = int(time.time() * 1000) % 9007199254740991  # Lấy millisecond từ epoch

        # Tạo dữ liệu sản phẩm (giả lập 1 sản phẩm)
        item = ItemData(name="Thanh toán đơn giản", quantity=1, price=payment.amount)

        # Tạo dữ liệu thanh toán
        payment_data = PaymentData(
            orderCode=order_code,
            amount=payment.amount,
            description=payment.description,
            items=[item],
            cancelUrl=f"{FRONTEND_URL}/payment/fail",
            returnUrl=f"{FRONTEND_URL}/payment/success"
        )

        # Gọi API payOS để tạo link thanh toán
        payment_link_response = payos.createPaymentLink(payment_data)
        return PaymentResponse(
            success=True,
            checkout_url=payment_link_response.checkoutUrl,
            order_code=payment_link_response.orderCode
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo liên kết thanh toán: {str(e)}")

# Endpoint kiểm tra trạng thái thanh toán
@router.post("/check-payment", response_model=PaymentStatusResponse)
async def check_payment(status_request: PaymentStatusRequest):
    try:
        # Kiểm tra trạng thái thanh toán từ payOS
        payment_info = payos.getPaymentLinkInformation(status_request.order_code)
        status = payment_info.status

        # Trả về kết quả đơn giản
        if status == "PAID":
            return PaymentStatusResponse(success=True, message="Thanh toán thành công")
        elif status == "CANCELLED":
            return PaymentStatusResponse(success=False, message="Thanh toán bị hủy")
        else:
            return PaymentStatusResponse(success=False, message="Thanh toán đang chờ xử lý")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi kiểm tra trạng thái: {str(e)}")