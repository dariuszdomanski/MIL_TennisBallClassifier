#ifndef ACCEPTHANDLER_HPP_
#define ACCEPTHANDLER_HPP_

#include <future>
#include <memory>

#include <nlohmann/json.hpp>
#include <scorpio/network.hpp>


class AcceptHandler: public MsgHandlerBase
{
public:
    AcceptHandler(const int& id, const std::shared_ptr<bool>& tennisBallDetection)
    :   MsgHandlerBase(id),
        tennisBallDetection_(tennisBallDetection)
    {}

    void handle(const nlohmann::json& msg, const int& grantorSocket) const override
    {
        auto sendFunction = [this, &grantorSocket] (bool& sendStatus, bool& lastDetection) {
            auto j = nlohmann::json();
            j["id"] = 1;
            j["TennisBall"] = *tennisBallDetection_;
            sendStatus = tcp::Sender::send(j, grantorSocket);
            lastDetection = *tennisBallDetection_;
        };
        handleLoopAsync_ = std::async([this, sendFunction] () {
            bool sendStatus = true;
            bool lastDetection = *tennisBallDetection_;
            sendFunction(sendStatus, lastDetection);
            while (sendStatus)
            {
                if (lastDetection != *tennisBallDetection_)
                {
                    sendFunction(sendStatus, lastDetection);
                }
            }
        });
    }

private:
    mutable std::shared_ptr<bool> tennisBallDetection_;
    mutable std::future<void> handleLoopAsync_;
};

#endif  // ACCEPTHANDLER_HPP_

