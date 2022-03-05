using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ViewLayer.Controllers;

namespace NUnitTestProject
{
    public class TestMethods
    {
        private readonly ItemController itemController;

        public TestMethods(ItemController itemController)
        {
            this.itemController = itemController;
        }

        public bool NegativePointsReplacement(string testNameIdentifier, string testNumberIdentifier)
        {
            return this.itemController.NegativePoints(testNameIdentifier, testNumberIdentifier);
        }
    }
}
