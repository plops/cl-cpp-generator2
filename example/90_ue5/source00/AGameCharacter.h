#ifndef AGAMECHARACTER_H
#define AGAMECHARACTER_H

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "Camera/CameraComponent.h" 
 
// the ...generated.h file must always be included last
 
#include "GameCharacter.generated.h" 
class AGameCharacter : public APawn {
        GENERATED_BODY()
        public:
         AGameCharacter ()       ;   
        protected:
        // Main Pawn Camera, https://docs.unrealengine.com/5.0/en-US/API/Runtime/Engine/Camera/UCameraComponent/
 
        virtual void BeginPlay ()     override  ;   
        UPROPERTY(EditAnywhere)
        UCameraComponent* Camera; 
        public:
        virtual void Tick (float DeltaTime)     override  ;   
        virtual void SetupPlayerInputComponent (class UInputComponent* PlayerInputComponent)     override  ;   
};

#endif /* !AGAMECHARACTER_H */