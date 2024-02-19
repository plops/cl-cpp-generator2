# Some documentation for RF_PHY

- this chinese blog post documents how to use RF_PHY https://www.cnblogs.com/azou/p/17918626.html

Here is the summary of the Chinese post in English (by Gemini Advanced 1.0): 

* The CH582, CH592, and CH32V208 are a series of low-power Bluetooth ICs that support 2.4 GHz wireless communication.
* The RF_PHY and RF_PHY_Hop examples are provided for these ICs.
* This post explains how to use these examples to implement a simple pairing mechanism.
* The pairing mechanism works as follows:
    1. Both devices use the default address (0x71764129) for initial communication.
    2. When the user triggers a pairing event, the sender sends its unique MAC address to the receiver.
    3. The receiver saves the sender's address and sends its own MAC address back to the sender.
    4. Both devices calculate a final communication address based on their MAC addresses.
    5. The devices are now paired and can communicate using the final address.
* This pairing mechanism can be used with both the RF_PHY and RF_PHY_Hop examples.
* The post also provides some tips for extending the pairing mechanism.

Here are some additional details about the pairing mechanism:

* The pairing mechanism is based on the assumption that both devices have the same basic configuration, such as the address and the hopping channel.
* The pairing mechanism can be easily modified to support different requirements. For example, you can use a different algorithm to calculate the final communication address.
* The pairing mechanism can be used to implement other features, such as authentication and encryption.

